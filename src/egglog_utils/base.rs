use crate::egglog_utils::api::*;
use crate::sort_fn;

/// Generate the egglog source equivalent to base.egg using the builder API.
pub fn base_egglog() -> String {
    let mut ctx = Program::new();

    // Rulesets
    ctx.ruleset("expr");
    ctx.ruleset("cleanup");
    ctx.ruleset("early");

    // Builtin sorts used as fields
    let int = ctx.builtin("i64");
    let float = ctx.builtin("f64");
    let string = ctx.builtin("String");

    // -------- SYMBOLIC ALGEBRA --------
    // Mutually recursive datatypes via (datatype* ...)
    let expression = ctx.class_star("Expression");
    let elist = ctx.class_star("EList");
    let dtype = ctx.class_star("DType");

    // Expression variants
    let num = sort_fn!(ctx, expression, MNum, value: int);
    let mfloat = sort_fn!(ctx, expression, MFloat, value: float);
    let iter = sort_fn!(ctx, expression, MIter);
    let m_var = sort_fn!(ctx, expression, MVar, name: string);
    let add = sort_fn!(ctx, expression, MAdd, a: expression, b: expression);
    let sub = sort_fn!(ctx, expression, MSub, a: expression, b: expression);
    let mul = sort_fn!(ctx, expression, MMul, a: expression, b: expression);
    let ceil_div = sort_fn!(ctx, expression, MCeilDiv, a: expression, b: expression);
    let div = sort_fn!(ctx, expression, MDiv, a: expression, b: expression);
    let modulo = sort_fn!(ctx, expression, MMod, a: expression, b: expression);
    let min = sort_fn!(ctx, expression, MMin, a: expression, b: expression);
    let max = sort_fn!(ctx, expression, MMax, a: expression, b: expression);
    let band = sort_fn!(ctx, expression, MAnd, a: expression, b: expression);
    let _or = sort_fn!(ctx, expression, MOr, a: expression, b: expression);
    let _gte = sort_fn!(ctx, expression, MGte, a: expression, b: expression);
    let _lt = sort_fn!(ctx, expression, MLt, a: expression, b: expression);
    let floor_to = sort_fn!(ctx, expression, MFloorTo, a: expression, b: expression);
    let replace =
        sort_fn!(ctx, expression, MReplace, x: expression, find: expression, replace: expression);

    // EList variants
    let cons = sort_fn!(ctx, elist, ECons, expr: expression, list: elist);
    let nil = sort_fn!(ctx, elist, ENil);
    let replace_list =
        sort_fn!(ctx, elist, MReplaceList, list: elist, from: expression, to: expression);
    let _replace_nth =
        sort_fn!(ctx, elist, ReplaceNthFromEnd, list: elist, to: expression, ind: int);
    let _remove_nth = sort_fn!(ctx, elist, RemoveNthFromEnd, list: elist, ind: int);
    let row_major = sort_fn!(ctx, elist, RowMajor, list: elist);

    // DType variants
    let _f32 = sort_fn!(ctx, dtype, F32);
    let _f16 = sort_fn!(ctx, dtype, F16);
    let _bf16 = sort_fn!(ctx, dtype, Bf16);
    let _int_dt = sort_fn!(ctx, dtype, Int);
    let _bool_dt = sort_fn!(ctx, dtype, Bool);

    // Shared pattern variables
    let a = pat("a");
    let b = pat("b");
    let c = pat("c");

    // ---- Algebraic rewrites ----

    // Commutativity of multiply
    ctx.rewrite(
        mul.call(a.clone(), b.clone()),
        mul.call(b.clone(), a.clone()),
        "expr",
    );

    // Addition constant folding: MAdd(MNum(a), MNum(b)) -> MNum(a+b) with union+subsume
    ctx.egglog_rule(
        &[
            eq(pat("?e"), add.call(num.call(pat("a")), num.call(pat("b")))),
            eq(pat("?ans"), iadd(pat("a"), pat("b"))),
        ],
        &[
            union(pat("?e"), num.call(pat("?ans"))),
            subsume(add.call(num.call(pat("a")), num.call(pat("b")))),
        ],
        "expr",
    );

    // Subtraction constant folding
    ctx.rewrite(
        sub.call(num.call(pat("a")), num.call(pat("b"))),
        num.call(isub(pat("a"), pat("b"))),
        "expr",
    );

    // Multiply constant folding with union+subsume
    ctx.egglog_rule(
        &[
            eq(
                pat("?e"),
                mul.call(num.call(pat("?a")), num.call(pat("?b"))),
            ),
            eq(pat("?prod"), imul(pat("?a"), pat("?b"))),
        ],
        &[
            union(pat("?e"), num.call(pat("?prod"))),
            subsume(mul.call(num.call(pat("?a")), num.call(pat("?b")))),
        ],
        "expr",
    );

    // Division constant folding
    ctx.rewrite_when(
        div.call(num.call(pat("a")), num.call(pat("b"))),
        num.call(idiv(pat("a"), pat("b"))),
        &[neq(i64(0), pat("b")), eq(i64(0), imod(pat("a"), pat("b")))],
        "expr",
    );

    // CeilDiv constant folding
    ctx.rewrite_when(
        ceil_div.call(num.call(pat("a")), num.call(pat("b"))),
        num.call(idiv(pat("a"), pat("b"))),
        &[neq(i64(0), pat("b")), eq(i64(0), imod(pat("a"), pat("b")))],
        "expr",
    );

    // Max/Min/And constant folding
    ctx.rewrite(
        max.call(num.call(pat("a")), num.call(pat("b"))),
        num.call(imax(pat("a"), pat("b"))),
        "expr",
    );
    ctx.rewrite(
        min.call(num.call(pat("a")), num.call(pat("b"))),
        num.call(imin(pat("a"), pat("b"))),
        "expr",
    );
    ctx.rewrite(
        band.call(num.call(pat("a")), num.call(pat("b"))),
        num.call(iand(pat("a"), pat("b"))),
        "expr",
    );

    // MFloat(-1.0) <-> MNum(-1)
    ctx.rewrite(mfloat.call(f64(-1.0)), num.call(i64(-1)), "expr");
    ctx.rewrite(num.call(i64(-1)), mfloat.call(f64(-1.0)), "expr");

    // Identity and zero rules
    ctx.rewrite(add.call(a.clone(), num.call(i64(0))), a.clone(), "expr");
    ctx.egglog_rule(
        &[eq(pat("?e"), mul.call(pat("?a"), num.call(i64(1))))],
        &[union(pat("?e"), pat("?a"))],
        "expr",
    );
    ctx.egglog_rule(
        &[eq(pat("?e"), mul.call(pat("?a"), num.call(i64(0))))],
        &[
            union(pat("?e"), num.call(i64(0))),
            subsume(mul.call(pat("?a"), num.call(i64(0)))),
        ],
        "expr",
    );
    ctx.rewrite(div.call(a.clone(), num.call(i64(1))), a.clone(), "expr");

    // Modulo rules
    let x = pat("?x");
    let y = pat("?y");
    ctx.rewrite(
        modulo.call(mul.call(x.clone(), y.clone()), y.clone()),
        num.call(i64(0)),
        "expr",
    );
    ctx.rewrite_when(
        modulo.call(
            modulo.call(x.clone(), num.call(pat("?y"))),
            num.call(pat("?z")),
        ),
        modulo.call(x.clone(), num.call(pat("?y"))),
        &[
            igte(pat("?z"), pat("?y")),
            eq(i64(0), imod(pat("?y"), pat("?z"))),
        ],
        "expr",
    );
    ctx.rewrite_when(
        modulo.call(
            modulo.call(x.clone(), num.call(pat("?y"))),
            num.call(pat("?z")),
        ),
        modulo.call(x.clone(), num.call(pat("?z"))),
        &[
            igte(pat("?y"), pat("?z")),
            eq(i64(0), imod(pat("?z"), pat("?y"))),
        ],
        "expr",
    );

    // Division rules
    ctx.rewrite(
        div.call(div.call(a.clone(), b.clone()), c.clone()),
        div.call(a.clone(), mul.call(b.clone(), c.clone())),
        "expr",
    );
    ctx.rewrite(
        add.call(div.call(a.clone(), b.clone()), c.clone()),
        div.call(
            add.call(a.clone(), mul.call(c.clone(), b.clone())),
            b.clone(),
        ),
        "expr",
    );

    // Add/Sub identity rules
    ctx.rewrite(
        add.call(a.clone(), sub.call(b.clone(), a.clone())),
        b.clone(),
        "expr",
    );
    ctx.rewrite(
        add.call(sub.call(b.clone(), a.clone()), a.clone()),
        b.clone(),
        "expr",
    );
    ctx.rewrite(sub.call(a.clone(), a.clone()), num.call(i64(0)), "expr");

    // Constant folding through add/sub associativity
    ctx.rewrite(
        add.call(
            sub.call(a.clone(), num.call(pat("?b"))),
            num.call(pat("?c")),
        ),
        sub.call(a.clone(), num.call(isub(pat("?b"), pat("?c")))),
        "expr",
    );
    ctx.rewrite(
        add.call(
            num.call(pat("?c")),
            sub.call(a.clone(), num.call(pat("?b"))),
        ),
        sub.call(a.clone(), num.call(isub(pat("?b"), pat("?c")))),
        "expr",
    );
    ctx.rewrite(
        sub.call(
            add.call(a.clone(), num.call(pat("?b"))),
            num.call(pat("?c")),
        ),
        add.call(a.clone(), num.call(isub(pat("?b"), pat("?c")))),
        "expr",
    );
    ctx.rewrite(
        sub.call(
            sub.call(a.clone(), num.call(pat("?b"))),
            num.call(pat("?c")),
        ),
        sub.call(a.clone(), num.call(iadd(pat("?b"), pat("?c")))),
        "expr",
    );

    // Factoring: a*b + a*c -> a*(b+c)
    ctx.rewrite(
        add.call(
            mul.call(a.clone(), b.clone()),
            mul.call(a.clone(), c.clone()),
        ),
        mul.call(a.clone(), add.call(b.clone(), c.clone())),
        "expr",
    );

    // a + a -> 2*a
    ctx.rewrite(
        add.call(a.clone(), a.clone()),
        mul.call(num.call(i64(2)), a.clone()),
        "expr",
    );

    // Constant folding through associativity: ((a + c1) + c2) -> (a + (c1+c2))
    ctx.egglog_rule(
        &[
            eq(
                pat("?e"),
                add.call(
                    add.call(pat("?a"), num.call(pat("?b"))),
                    num.call(pat("?c")),
                ),
            ),
            eq(pat("?ans"), iadd(pat("?b"), pat("?c"))),
        ],
        &[
            union(pat("?e"), add.call(pat("?a"), num.call(pat("?ans")))),
            subsume(add.call(
                add.call(pat("?a"), num.call(pat("?b"))),
                num.call(pat("?c")),
            )),
        ],
        "expr",
    );
    ctx.rewrite(
        add.call(
            add.call(num.call(pat("?b")), m_var.call(pat("?v"))),
            num.call(pat("?c")),
        ),
        add.call(m_var.call(pat("?v")), num.call(iadd(pat("?b"), pat("?c")))),
        "expr",
    );
    ctx.rewrite(
        add.call(
            add.call(num.call(pat("?b")), mul.call(pat("?n"), a.clone())),
            num.call(pat("?c")),
        ),
        add.call(
            mul.call(pat("?n"), a.clone()),
            num.call(iadd(pat("?b"), pat("?c"))),
        ),
        "expr",
    );

    // Combine like terms: (n*a) + a -> (n+1)*a
    ctx.rewrite_subsume(
        add.call(mul.call(num.call(pat("?n")), pat("?a")), pat("?a")),
        mul.call(num.call(iadd(pat("?n"), i64(1))), pat("?a")),
        "expr",
    );
    ctx.rewrite_subsume(
        add.call(pat("?a"), mul.call(num.call(pat("?n")), pat("?a"))),
        mul.call(num.call(iadd(pat("?n"), i64(1))), pat("?a")),
        "expr",
    );
    ctx.rewrite_subsume(
        add.call(mul.call(pat("?a"), num.call(pat("?n"))), pat("?a")),
        mul.call(num.call(iadd(pat("?n"), i64(1))), pat("?a")),
        "expr",
    );
    ctx.rewrite_subsume(
        add.call(pat("?a"), mul.call(pat("?a"), num.call(pat("?n")))),
        mul.call(num.call(iadd(pat("?n"), i64(1))), pat("?a")),
        "expr",
    );

    // Combine repeated variables: ((a + v) + v) -> (a + 2*v)
    ctx.rewrite_subsume(
        add.call(
            add.call(pat("?a"), m_var.call(pat("?v"))),
            m_var.call(pat("?v")),
        ),
        add.call(pat("?a"), mul.call(num.call(i64(2)), m_var.call(pat("?v")))),
        "expr",
    );
    ctx.rewrite_subsume(
        add.call(
            add.call(m_var.call(pat("?v")), pat("?a")),
            m_var.call(pat("?v")),
        ),
        add.call(pat("?a"), mul.call(num.call(i64(2)), m_var.call(pat("?v")))),
        "expr",
    );

    // Accumulate: ((n*a + b) + a) -> ((n+1)*a + b)
    ctx.rewrite_subsume(
        add.call(
            add.call(mul.call(num.call(pat("?n")), pat("?a")), pat("?b")),
            pat("?a"),
        ),
        add.call(
            mul.call(num.call(iadd(pat("?n"), i64(1))), pat("?a")),
            pat("?b"),
        ),
        "expr",
    );
    ctx.rewrite_subsume(
        add.call(
            add.call(pat("?b"), mul.call(num.call(pat("?n")), pat("?a"))),
            pat("?a"),
        ),
        add.call(
            pat("?b"),
            mul.call(num.call(iadd(pat("?n"), i64(1))), pat("?a")),
        ),
        "expr",
    );

    // ---- Replacement over expressions ----
    ctx.rewrite_when(
        replace.call(pat("?x"), pat("?y"), pat("?z")),
        pat("?z"),
        &[eq(pat("?x"), pat("?y"))],
        "expr",
    );

    // MReplace distributes over binary ops
    macro_rules! replace_over {
        ($op:expr) => {
            ctx.rewrite(
                replace.call($op.call(pat("?a"), pat("?b")), pat("?x"), pat("?y")),
                $op.call(
                    replace.call(pat("?a"), pat("?x"), pat("?y")),
                    replace.call(pat("?b"), pat("?x"), pat("?y")),
                ),
                "expr",
            );
        };
    }
    replace_over!(add);
    replace_over!(sub);
    replace_over!(mul);
    replace_over!(div);
    replace_over!(ceil_div);
    replace_over!(modulo);
    replace_over!(min);
    replace_over!(max);
    replace_over!(floor_to);

    ctx.rewrite(
        replace.call(num.call(pat("?n")), pat("?x"), pat("?y")),
        num.call(pat("?n")),
        "expr",
    );
    ctx.rewrite_when(
        replace.call(m_var.call(pat("?z")), pat("?find"), pat("?replace")),
        m_var.call(pat("?z")),
        &[neq(pat("?find"), m_var.call(pat("?z")))],
        "expr",
    );
    ctx.rewrite_when(
        replace.call(iter.call(), pat("?find"), pat("?replace")),
        iter.call(),
        &[neq(pat("?find"), iter.call())],
        "expr",
    );

    // ---- EList helper functions ----

    // len
    ctx.function_decl("len", &[elist], int, "new");
    ctx.egglog_rule(
        &[eq(pat("?e"), nil.call())],
        &[set(prim("len", vec![pat("?e")]), i64(0))],
        "expr",
    );
    ctx.egglog_rule(
        &[
            eq(pat("?e"), cons.call(pat("?expr"), pat("?list"))),
            eq(pat("?prev_len"), prim("len", vec![pat("?list")])),
        ],
        &[set(
            prim("len", vec![pat("?e")]),
            iadd(pat("?prev_len"), i64(1)),
        )],
        "expr",
    );

    // nth_from_end
    ctx.function_decl("nth_from_end", &[elist, int], expression, "new");
    ctx.egglog_rule(
        &[
            eq(pat("?e"), cons.call(pat("?expr"), pat("?list"))),
            eq(pat("?list_len"), prim("len", vec![pat("?list")])),
        ],
        &[set(
            prim("nth_from_end", vec![pat("?e"), pat("?list_len")]),
            pat("?expr"),
        )],
        "expr",
    );
    ctx.egglog_rule(
        &[
            eq(pat("?e"), cons.call(pat("?expr"), pat("?list"))),
            eq(
                pat("?other_nth"),
                prim("nth_from_end", vec![pat("?list"), pat("?n")]),
            ),
        ],
        &[set(
            prim("nth_from_end", vec![pat("?e"), pat("?n")]),
            pat("?other_nth"),
        )],
        "expr",
    );

    // n_elements
    ctx.function_decl("n_elements", &[elist], expression, "new");
    ctx.egglog_rule(
        &[eq(pat("?e"), nil.call())],
        &[set(prim("n_elements", vec![pat("?e")]), num.call(i64(1)))],
        "expr",
    );
    ctx.egglog_rule(
        &[
            eq(pat("?e"), cons.call(pat("?dim"), pat("?other"))),
            eq(pat("?other_elems"), prim("n_elements", vec![pat("?other")])),
        ],
        &[set(
            prim("n_elements", vec![pat("?e")]),
            mul.call(pat("?dim"), pat("?other_elems")),
        )],
        "expr",
    );

    // RowMajor recursive case
    ctx.egglog_rule(
        &[
            eq(
                pat("?other"),
                cons.call(pat("?other_dim"), pat("?other_other")),
            ),
            eq(pat("?list"), cons.call(pat("?d"), pat("?other"))),
            eq(pat("?e"), row_major.call(pat("?list"))),
            eq(pat("?n_elems"), prim("n_elements", vec![pat("?other")])),
        ],
        &[union(
            pat("?e"),
            cons.call(pat("?n_elems"), row_major.call(pat("?other"))),
        )],
        "expr",
    );

    // RowMajor base case
    ctx.rewrite(
        row_major.call(cons.call(pat("?dim"), nil.call())),
        cons.call(num.call(i64(1)), nil.call()),
        "expr",
    );

    // MReplaceList
    ctx.rewrite(
        replace_list.call(
            cons.call(pat("?expr"), pat("?list")),
            pat("?from"),
            pat("?to"),
        ),
        cons.call(
            replace.call(pat("?expr"), pat("?from"), pat("?to")),
            replace_list.call(pat("?list"), pat("?from"), pat("?to")),
        ),
        "expr",
    );

    // ReplaceNthFromEnd rules
    ctx.egglog_rule(
        &[
            eq(
                pat("?e"),
                prim(
                    "ReplaceNthFromEnd",
                    vec![
                        cons.call(pat("?expr"), pat("?list")),
                        pat("?to"),
                        pat("?ind"),
                    ],
                ),
            ),
            eq(pat("?ind"), prim("len", vec![pat("?list")])),
        ],
        &[union(pat("?e"), cons.call(pat("?to"), pat("?list")))],
        "expr",
    );
    ctx.egglog_rule(
        &[
            eq(
                pat("?e"),
                prim(
                    "ReplaceNthFromEnd",
                    vec![
                        cons.call(pat("?expr"), pat("?list")),
                        pat("?to"),
                        pat("?ind"),
                    ],
                ),
            ),
            ilt(pat("?ind"), prim("len", vec![pat("?list")])),
        ],
        &[union(
            pat("?e"),
            cons.call(
                pat("?expr"),
                prim(
                    "ReplaceNthFromEnd",
                    vec![pat("?list"), pat("?to"), pat("?ind")],
                ),
            ),
        )],
        "expr",
    );

    // RemoveNthFromEnd rules
    ctx.egglog_rule(
        &[
            eq(
                pat("?e"),
                prim(
                    "RemoveNthFromEnd",
                    vec![cons.call(pat("?expr"), pat("?list")), pat("?ind")],
                ),
            ),
            eq(pat("?ind"), prim("len", vec![pat("?list")])),
        ],
        &[union(pat("?e"), pat("?list"))],
        "expr",
    );
    ctx.egglog_rule(
        &[
            eq(
                pat("?e"),
                prim(
                    "RemoveNthFromEnd",
                    vec![cons.call(pat("?expr"), pat("?list")), pat("?ind")],
                ),
            ),
            ilt(pat("?ind"), prim("len", vec![pat("?list")])),
        ],
        &[union(
            pat("?e"),
            cons.call(
                pat("?expr"),
                prim("RemoveNthFromEnd", vec![pat("?list"), pat("?ind")]),
            ),
        )],
        "expr",
    );

    ctx.to_egglog_string()
}
