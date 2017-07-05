// This file is a part of Julia. License is MIT: https://julialang.org/license

#define DEBUG_TYPE "alloc_opt"
#undef DEBUG
#include "llvm-version.h"

#include <llvm/IR/Value.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>

#include "fix_llvm_assert.h"

#include "codegen_shared.h"
#include "julia.h"
#include "julia_internal.h"

#include <map>
#include <set>

using namespace llvm;

extern std::pair<MDNode*,MDNode*> tbaa_make_child(const char *name, MDNode *parent=nullptr, bool isConstant=false);

namespace {

static void copyMetadata(Instruction *dest, const Instruction *src)
{
#if JL_LLVM_VERSION < 40000
    if (!src->hasMetadata())
        return;
    SmallVector<std::pair<unsigned,MDNode*>,4> TheMDs;
    src->getAllMetadataOtherThanDebugLoc(TheMDs);
    for (const auto &MD : TheMDs)
        dest->setMetadata(MD.first, MD.second);
    dest->setDebugLoc(src->getDebugLoc());
#else
    dest->copyMetadata(*src);
#endif
}

/**
 * Promote `julia.gc_alloc_obj` which do not have escaping root to a alloca and
 * lower other ones to real GC allocation.
 * Uses that are not considerer to escape the object (i.e. heap address) includes,
 *
 * * load
 * * `pointer_from_objref`
 * * `ccall` gcroot array (`jl_roots` operand bundle)
 * * store (as address)
 * * addrspacecast, bitcast, getelementptr
 *
 *     The results of these cast instructions will be scanned recursively.
 *
 * All other uses are considered escape conservatively.
 */

struct AllocOpt : public FunctionPass {
    static char ID;
    AllocOpt(bool opt=true)
        : FunctionPass(ID),
          optimize(opt)
    {}

private:
    bool optimize;
    LLVMContext *ctx;

    const DataLayout *DL;

    Function *alloc_obj;
    Function *pool_alloc;
    Function *big_alloc;
    Function *ptr_from_objref;
    Function *lifetime_start;
    Function *lifetime_end;

    Type *T_int8;
    Type *T_int32;
    Type *T_int64;
    Type *T_size;
    Type *T_pint8;
    Type *T_prjlvalue;
    Type *T_pjlvalue;
    Type *T_pjlvalue_der;
    Type *T_pprjlvalue;
    Type *T_ppjlvalue_der;

    MDNode *tbaa_tag;

    struct CheckInstFrame {
        Instruction *parent;
        size_t offset;
        // TODO switch to use_iterator
        Instruction::user_iterator user_it;
        Instruction::user_iterator user_end;
    };
    typedef SmallVector<CheckInstFrame, 4> CheckInstStack;
    struct ReplaceUsesFrame {
        Instruction *orig_i;
        Instruction *new_i;
        // TODO switch to loop on `user_empty()`
        SmallVector<User*,4> users;
        size_t idx;
        ReplaceUsesFrame(Instruction *orig_i, Instruction *new_i)
            : orig_i(orig_i),
              new_i(new_i),
              users(orig_i->user_begin(), orig_i->user_end()),
              idx(0)
        {}
    };
    typedef SmallVector<ReplaceUsesFrame,4> ReplaceUsesStack;

    struct LifetimeMarker {
        LifetimeMarker(AllocOpt &pass)
            : pass(pass),
              first_safepoint{},
              stack{}
        {}
        // insert llvm.lifetime.* calls for `ptr` with size `sz`
        // based on the use of `orig` given in `alloc_uses`.
        void insert(Instruction *ptr, Constant *sz, Instruction *orig,
                    const std::set<Instruction*> &alloc_uses);
    private:
        Instruction *getFirstSafepoint(BasicBlock *bb);
        void insertEnd(Instruction *ptr, Constant *sz, Instruction *insert);
        struct Frame {
            BasicBlock *bb;
            pred_iterator p_cur;
            pred_iterator p_end;
            Frame(BasicBlock *bb)
                : bb(bb),
                  p_cur(pred_begin(bb)),
                  p_end(pred_end(bb))
            {}
        };
        AllocOpt &pass;
        std::map<BasicBlock*,Instruction*> first_safepoint;
        SmallVector<Frame,4> stack;
    };

    bool doInitialization(Module &m) override;
    bool runOnFunction(Function &F) override;
    bool checkInst(Instruction *I, CheckInstStack &stack, std::set<Instruction*> &uses,
                   bool &ignore_tag);
    void replaceUsesWith(Instruction *orig_i, Instruction *new_i, ReplaceUsesStack &stack);
    void lowerAlloc(CallInst *I, size_t sz);
    bool isSafepoint(Instruction *inst);
    void getAnalysisUsage(AnalysisUsage &AU) const override
    {
        if (optimize) {
            FunctionPass::getAnalysisUsage(AU);
            AU.addRequired<DominatorTreeWrapperPass>();
            AU.addPreserved<DominatorTreeWrapperPass>();
            AU.setPreservesCFG();
        }
    }
};

Instruction *AllocOpt::LifetimeMarker::getFirstSafepoint(BasicBlock *bb)
{
    auto it = first_safepoint.find(bb);
    if (it != first_safepoint.end())
        return it->second;
    Instruction *first = nullptr;
    for (auto &I: *bb) {
        if (pass.isSafepoint(&I)) {
            first = &I;
            break;
        }
    }
    first_safepoint[bb] = first;
    return first;
}

void AllocOpt::LifetimeMarker::insertEnd(Instruction *ptr, Constant *sz, Instruction *insert)
{
    BasicBlock::iterator it(insert);
    BasicBlock::iterator begin(insert->getParent()->begin());
    // Makes sure that the end is inserted before nearby start.
    // We insert start before the allocation call, if it is the first safepoint we find for
    // another instruction, it's better if we insert the end before the start instead of the
    // allocation so that the two allocation do not have overlapping lifetime.
    while (it != begin) {
        --it;
        if (auto II = dyn_cast<IntrinsicInst>(&*it)) {
            if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                II->getIntrinsicID() == Intrinsic::lifetime_end) {
                insert = II;
                continue;
            }
        }
        break;
    }
    CallInst::Create(pass.lifetime_end, {sz, ptr}, "", insert);
}

void AllocOpt::LifetimeMarker::insert(Instruction *ptr, Constant *sz, Instruction *orig,
                                      const std::set<Instruction*> &alloc_uses)
{
    CallInst::Create(pass.lifetime_start, {sz, ptr}, "", orig);
    BasicBlock *def_bb = orig->getParent();
    std::set<BasicBlock*> bbs{def_bb};
    auto &DT = pass.getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    // Collect all BB where the allocation is live
    for (auto use: alloc_uses) {
        auto bb = use->getParent();
        if (!bbs.insert(bb).second)
            continue;
        assert(stack.empty());
        Frame cur{bb};
        while (true) {
            assert(cur.p_cur != cur.p_end);
            auto pred = *cur.p_cur;
            ++cur.p_cur;
            if (bbs.insert(pred).second) {
                if (cur.p_cur != cur.p_end)
                    stack.push_back(cur);
                cur = Frame(pred);
            }
            if (cur.p_cur == cur.p_end) {
                if (stack.empty())
                    break;
                cur = stack.back();
                stack.pop_back();
            }
        }
    }
#ifndef JL_NDEBUG
    for (auto bb: bbs) {
        if (bb == def_bb)
            continue;
        if (DT.dominates(orig, bb))
            continue;
        auto F = bb->getParent();
#if JL_LLVM_VERSION >= 50000
        F->print(llvm::dbgs(), nullptr, false, true);
        orig->print(llvm::dbgs(), true);
        jl_safe_printf("Does not dominate BB:\n");
        bb->print(llvm::dbgs(), true);
#else
        F->dump();
        orig->dump();
        jl_safe_printf("Does not dominate BB:\n");
        bb->dump();
#endif
        abort();
    }
#endif
    // For each BB, find the first instruction(s) where the allocation is possibly dead.
    // If all successors are live, then there isn't one.
    // If all successors are dead, then it's the first instruction after the last use
    // within the BB.
    // If some successors are live and others are dead, it's the first instruction in
    // the successors that are dead.
    std::vector<Instruction*> first_dead;
    for (auto bb: bbs) {
        bool has_use = false;
        for (auto succ: successors(bb)) {
            // def_bb is the only bb in bbs that's not dominated by orig
            if (succ != def_bb && bbs.find(succ) != bbs.end()) {
                has_use = true;
                break;
            }
        }
        if (has_use) {
            for (auto succ: successors(bb)) {
                if (bbs.find(succ) == bbs.end()) {
                    first_dead.push_back(&*succ->begin());
                }
            }
        }
        else {
            for (auto it = bb->rbegin(), end = bb->rend(); it != end; ++it) {
                if (alloc_uses.find(&*it) != alloc_uses.end()) {
                    --it;
                    first_dead.push_back(&*it);
                    break;
                }
            }
        }
    }
    bbs.clear();
    // There can/need only be one lifetime.end for each allocation in each bb, use bbs
    // to record that.
    // Iterate through the first dead and find the first safepoint following each of them.
    while (!first_dead.empty()) {
        auto I = first_dead.back();
        first_dead.pop_back();
        auto bb = I->getParent();
        if (!bbs.insert(bb).second)
            continue;
        if (I == &*bb->begin()) {
            // There's no use in or after this bb. If this bb is not dominated by
            // the def then it has to be dead on entering this bb.
            // Otherwise, there could be use that we don't track
            // before hitting the next safepoint.
            if (!DT.dominates(orig, bb)) {
                insertEnd(ptr, sz, &*bb->getFirstInsertionPt());
                continue;
            }
            else if (auto insert = getFirstSafepoint(bb)) {
                insertEnd(ptr, sz, insert);
            }
        }
        else {
            assert(bb == def_bb || DT.dominates(orig, I));
            BasicBlock::iterator it(I);
            BasicBlock::iterator end = bb->end();
            bool safepoint_found = false;
            for (; it != end; ++it) {
                auto insert = &*it;
                if (pass.isSafepoint(insert)) {
                    insertEnd(ptr, sz, insert);
                    safepoint_found = true;
                    break;
                }
            }
            if (safepoint_found) {
                continue;
            }
        }
        for (auto succ: successors(bb)) {
            first_dead.push_back(&*succ->begin());
        }
    }
}

static void addRetNoAlias(Function *F)
{
#if JL_LLVM_VERSION >= 50000
    F->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
#else
    F->addAttribute(AttributeSet::ReturnIndex, Attribute::NoAlias);
#endif
}

bool AllocOpt::doInitialization(Module &M)
{
    ctx = &M.getContext();
    DL = &M.getDataLayout();

    alloc_obj = M.getFunction("julia.gc_alloc_obj");
    if (!alloc_obj)
        return false;

    ptr_from_objref = M.getFunction("julia.pointer_from_objref");

    T_prjlvalue = alloc_obj->getReturnType();
    T_pjlvalue = PointerType::get(cast<PointerType>(T_prjlvalue)->getElementType(), 0);
    T_pjlvalue_der = PointerType::get(cast<PointerType>(T_prjlvalue)->getElementType(),
                                      AddressSpace::Derived);
    T_pprjlvalue = PointerType::get(T_prjlvalue, 0);
    T_ppjlvalue_der = PointerType::get(T_prjlvalue, AddressSpace::Derived);

    T_int8 = Type::getInt8Ty(*ctx);
    T_int32 = Type::getInt32Ty(*ctx);
    T_int64 = Type::getInt64Ty(*ctx);
    T_size = sizeof(void*) == 8 ? T_int64 : T_int32;
    T_pint8 = PointerType::get(T_int8, 0);

    if (!(pool_alloc = M.getFunction("jl_gc_pool_alloc"))) {
        std::vector<Type*> alloc_pool_args(0);
        alloc_pool_args.push_back(T_pint8);
        alloc_pool_args.push_back(T_int32);
        alloc_pool_args.push_back(T_int32);
        pool_alloc = Function::Create(FunctionType::get(T_prjlvalue, alloc_pool_args, false),
                                      Function::ExternalLinkage, "jl_gc_pool_alloc", &M);
        addRetNoAlias(pool_alloc);
    }
    if (!(big_alloc = M.getFunction("jl_gc_big_alloc"))) {
        std::vector<Type*> alloc_big_args(0);
        alloc_big_args.push_back(T_pint8);
        alloc_big_args.push_back(T_size);
        big_alloc = Function::Create(FunctionType::get(T_prjlvalue, alloc_big_args, false),
                                     Function::ExternalLinkage, "jl_gc_big_alloc", &M);
        addRetNoAlias(big_alloc);
    }

#if JL_LLVM_VERSION >= 50000
    lifetime_start = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_start, { T_pint8 });
    lifetime_end = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_end, { T_pint8 });
#else
    lifetime_start = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_start);
    lifetime_end = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_end);
#endif

    MDNode *tbaa_data;
    MDNode *tbaa_data_scalar;
    std::tie(tbaa_data, tbaa_data_scalar) = tbaa_make_child("jtbaa_data");
    tbaa_tag = tbaa_make_child("jtbaa_tag", tbaa_data_scalar).first;

    return true;
}

bool AllocOpt::checkInst(Instruction *I, CheckInstStack &stack, std::set<Instruction*> &uses,
                         bool &ignore_tag)
{
    uses.clear();
    if (I->user_empty())
        return true;
    CheckInstFrame cur{I, 0, I->user_begin(), I->user_end()};
    stack.clear();

    // Recursion
    auto push_inst = [&] (Instruction *inst) {
        if (cur.user_it != cur.user_end)
            stack.push_back(cur);
        cur.parent = inst;
        cur.user_it = inst->user_begin();
        cur.user_end = inst->user_end();
    };

    auto check_inst = [&] (Instruction *inst) {
        if (isa<LoadInst>(inst))
            return true;
        if (auto call = dyn_cast<CallInst>(inst)) {
            // TODO: on LLVM 5.0 we may need to handle certain llvm intrinsics
            // including `memcpy`, `memset` etc. We might also need to handle
            // `memcmp` by coverting to our own intrinsic and lower it after the gc root pass.
            if (ptr_from_objref && ptr_from_objref == call->getCalledFunction())
                return true;
            // Only use in argument counts, use in operand bundle doesn't since it cannot escape.
            for (auto &arg: call->arg_operands()) {
                if (dyn_cast<Instruction>(&arg) == cur.parent) {
                    return false;
                }
            }
            if (call->getNumOperandBundles() != 1)
                return false;
            auto obuse = call->getOperandBundleAt(0);
            if (obuse.getTagName() != "jl_roots")
                return false;
            return true;
        }
        if (auto store = dyn_cast<StoreInst>(inst)) {
            auto storev = store->getValueOperand();
            // Only store value count
            if (storev == cur.parent)
                return false;
            // There's GC root in this object.
            if (auto ptrtype = dyn_cast<PointerType>(storev->getType())) {
                if (ptrtype->getAddressSpace() == AddressSpace::Tracked) {
                    return false;
                }
            }
            return true;
        }
        if (isa<AddrSpaceCastInst>(inst) || isa<BitCastInst>(inst)) {
            push_inst(inst);
            return true;
        }
        if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
            APInt apoffset(sizeof(void*) * 8, cur.offset, true);
            if (ignore_tag && (!gep->accumulateConstantOffset(*DL, apoffset) ||
                               apoffset.isNegative()))
                ignore_tag = false;
            push_inst(inst);
            cur.offset = apoffset.getLimitedValue();
            // Overflow?
            if (cur.offset == UINT64_MAX)
                ignore_tag = false;
            return true;
        }
        return false;
    };

    while (true) {
        assert(cur.user_it != cur.user_end);
        auto user = *cur.user_it;
        auto inst = dyn_cast<Instruction>(user);
        ++cur.user_it;
        if (!inst)
            return false;
        if (!check_inst(inst))
            return false;
        uses.insert(inst);
        if (cur.user_it == cur.user_end) {
            if (stack.empty())
                return true;
            cur = stack.back();
            stack.pop_back();
        }
    }
}

// This function needs to handle all cases `AllocOpt::checkInst` can handle.
// This function should not erase any safepoint so that the lifetime marker can find and cache
// all the original safepoints.
void AllocOpt::replaceUsesWith(Instruction *orig_inst, Instruction *new_inst,
                               ReplaceUsesStack &stack)
{
    auto simple_replace = [&] (Instruction *orig_i, Instruction *new_i) {
        if (orig_i->user_empty()) {
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        Type *orig_t = orig_i->getType();
        Type *new_t = new_i->getType();
        if (orig_t == new_t) {
            orig_i->replaceAllUsesWith(new_i);
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        return false;
    };
    if (simple_replace(orig_inst, new_inst))
        return;
    assert(stack.empty());
    stack.emplace_back(orig_inst, new_inst);
    ReplaceUsesFrame *cur = &stack[0];
    auto finish_cur = [&] () {
        assert(cur->orig_i->user_empty());
        if (cur->orig_i != orig_inst) {
            cur->orig_i->eraseFromParent();
        }
    };
    auto push_frame = [&] (Instruction *orig_i, Instruction *new_i) {
        if (simple_replace(orig_i, new_i))
            return;
        stack.emplace_back(orig_i, new_i);
        cur = &stack.back();
    };
    // Both `orig_i` and `new_i` should be pointer of the same type
    // but possibly different address spaces. `new_i` is always in addrspace 0.
    auto replace_inst = [&] (Instruction *user) {
        Instruction *orig_i = cur->orig_i;
        Instruction *new_i = cur->new_i;
        if (isa<LoadInst>(user) || isa<StoreInst>(user)) {
            user->replaceUsesOfWith(orig_i, new_i);
        }
        else if (auto call = dyn_cast<CallInst>(user)) {
            if (ptr_from_objref && ptr_from_objref == call->getCalledFunction()) {
                call->replaceAllUsesWith(new_i);
                call->eraseFromParent();
                return;
            }
            // remove from operand bundle
            Type *new_t = new_i->getType();
            user->replaceUsesOfWith(orig_i, ConstantPointerNull::get(cast<PointerType>(new_t)));
        }
        else if (isa<AddrSpaceCastInst>(user) || isa<BitCastInst>(user)) {
            auto cast_t = PointerType::get(cast<PointerType>(user->getType())->getElementType(),
                                           0);
            auto replace_i = new_i;
            Type *orig_t = orig_i->getType();
            if (cast_t != orig_t) {
                replace_i = new BitCastInst(replace_i, cast_t, "", user);
                replace_i->setDebugLoc(user->getDebugLoc());
                replace_i->takeName(user);
            }
            push_frame(user, replace_i);
        }
        else if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
            SmallVector<Value *, 4> IdxOperands(gep->idx_begin(), gep->idx_end());
            auto new_gep = GetElementPtrInst::Create(gep->getSourceElementType(),
                                                     new_i, IdxOperands,
                                                     gep->getName(), gep);
            new_gep->setIsInBounds(gep->isInBounds());
            new_gep->takeName(gep);
            copyMetadata(new_gep, gep);
            push_frame(gep, new_gep);
        }
        else {
            abort();
        }
    };

    while (true) {
        assert(cur->idx < cur->users.size());
        auto user = cast<Instruction>(cur->users[cur->idx]);
        cur->idx++;
        replace_inst(user);
        while (cur->idx >= cur->users.size()) {
            finish_cur();
            stack.pop_back();
            if (stack.empty())
                return;
            cur = &stack.back();
        }
    }
}

void AllocOpt::lowerAlloc(CallInst *I, size_t sz)
{
    // TODO use IRBuilder
    int osize;
    int offset = jl_gc_classify_pools(sz, &osize);
    auto ptls = I->getArgOperand(0);
    CallInst *newI;
    if (offset < 0) {
        newI = CallInst::Create(big_alloc, {ptls, ConstantInt::get(T_size, sz + sizeof(void*))},
                                None, "", I);
    }
    else {
        auto pool_offs = ConstantInt::get(T_int32, offset);
        auto pool_osize = ConstantInt::get(T_int32, osize);
        newI = CallInst::Create(pool_alloc, {ptls, pool_offs, pool_osize}, None, "", I);
    }
    auto tag = I->getArgOperand(2);
    newI->setAttributes(I->getAttributes());
    copyMetadata(newI, I);
    const auto &dbg = I->getDebugLoc();
    auto derived = new AddrSpaceCastInst(newI, T_pjlvalue_der, "", I);
    derived->setDebugLoc(dbg);
    auto cast = new BitCastInst(derived, T_ppjlvalue_der, "", I);
    cast->setDebugLoc(dbg);
    auto tagaddr = GetElementPtrInst::Create(T_prjlvalue, cast, {ConstantInt::get(T_size, -1)},
                                             "", I);
    tagaddr->setDebugLoc(dbg);
    auto store = new StoreInst(tag, tagaddr, I);
    store->setMetadata(LLVMContext::MD_tbaa, tbaa_tag);
    store->setDebugLoc(dbg);
    I->replaceAllUsesWith(newI);
}

bool AllocOpt::isSafepoint(Instruction *inst)
{
    auto call = dyn_cast<CallInst>(inst);
    if (!call)
        return false;
    if (isa<IntrinsicInst>(call))
        return false;
    if (auto callee = call->getCalledFunction()) {
        // Known functions emitted in codegen that are not safepoints
        if (callee == ptr_from_objref || callee->getName() == "memcmp") {
            return false;
        }
    }
    return true;
}

bool AllocOpt::runOnFunction(Function &F)
{
    if (!alloc_obj)
        return false;
    std::map<CallInst*,size_t> allocs;
    for (auto &bb: F) {
        for (auto &I: bb) {
            auto call = dyn_cast<CallInst>(&I);
            if (!call)
                continue;
            auto callee = call->getCalledFunction();
            if (!callee)
                continue;
            size_t sz;
            if (callee == alloc_obj) {
                assert(call->getNumArgOperands() == 3);
                sz = (size_t)cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
            }
            else {
                continue;
            }
            allocs[call] = sz;
        }
    }

    auto &entry = F.getEntryBlock();
    auto first = &entry.front();
    CheckInstStack check_stack;
    ReplaceUsesStack replace_stack;
    std::set<Instruction*> alloc_uses;
    LifetimeMarker lifetime(*this);
    for (auto it: allocs) {
        bool ignore_tag = true;
        auto orig = it.first;
        if (optimize && checkInst(orig, check_stack, alloc_uses, ignore_tag)) {
            // The allocation does not escape or get used in a phi node so none of the derived
            // SSA from it are live when we run the allocation again.
            // It is now safe to promote the allocation to an entry block alloca.
            size_t sz = it.second;
            size_t align = 1;
            // TODO make codegen handling of alignment consistent and pass that as a parameter
            // to the allocation function directly.
            if (!ignore_tag) {
                align = sz <= 8 ? 8 : 16;
                sz += align;
            }
            else if (sz >= 16) {
                align = 16;
            }
            else if (sz >= 8) {
                align = 8;
            }
            else if (sz >= 4) {
                align = 4;
            }
            else if (sz >= 2) {
                align = 2;
            }
            // No debug info for prolog instructions
#if JL_LLVM_VERSION >= 50000
            // TODO try alloca with intn
            Instruction *ptr = new AllocaInst(T_int8, 0, ConstantInt::get(T_int32, sz),
                                              align, "", first);
#else
            Instruction *ptr = new AllocaInst(T_int8, ConstantInt::get(T_int32, sz),
                                              align, "", first);
#endif
            lifetime.insert(ptr, ConstantInt::get(T_int64, sz), orig, alloc_uses);
            // Someone might be reading the tag, initialize it.
            if (!ignore_tag) {
                ptr = GetElementPtrInst::CreateInBounds(T_int8, ptr,
                                                        {ConstantInt::get(T_int32, align)}, "",
                                                        first);
                auto cast = new BitCastInst(ptr, T_pprjlvalue, "", first);
                auto tagaddr = GetElementPtrInst::Create(T_prjlvalue, cast,
                                                         {ConstantInt::get(T_int32, -1)},
                                                         "", first);
                auto tag = orig->getArgOperand(2);
                auto store = new StoreInst(tag, tagaddr, orig);
                store->setMetadata(LLVMContext::MD_tbaa, tbaa_tag);
                store->setDebugLoc(orig->getDebugLoc());
            }
            auto cast = new BitCastInst(ptr, T_pjlvalue, "", first);
            replaceUsesWith(orig, cast, replace_stack);
        }
        else {
            lowerAlloc(orig, it.second);
        }
    }
    for (auto it: allocs)
        it.first->eraseFromParent();
    return true;
}

char AllocOpt::ID = 0;
static RegisterPass<AllocOpt> X("AllocOpt", "Promote heap allocation to stack",
                                false /* Only looks at CFG */,
                                false /* Analysis Pass */);

}

Pass *createAllocOptPass(bool opt)
{
    return new AllocOpt(opt);
}
