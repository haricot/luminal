#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SortClassId(pub(crate) usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SortId<const N: usize = 0>(pub(crate) usize);

pub type AnySortId = SortId<0>;

impl<const N: usize> SortId<N> {
    pub fn raw(self) -> usize {
        self.0
    }

    pub fn erase(self) -> AnySortId {
        SortId::<0>(self.0)
    }
}

/// Defines a variant and returns a struct whose `.call()` method has properly named parameters.
///
/// ```ignore
/// let add = sort_fn!(ctx, expr, add, lhs: expr, rhs: expr);
/// add.call(my_lhs, my_rhs)  // IDE shows: fn call(&self, lhs: Term, rhs: Term) -> Term
/// ```
#[macro_export]
macro_rules! sort_fn {
    ($ctx:expr, $class:expr, $name:ident $(, $arg:ident : $sort:expr)* $(,)?) => {{
        #[allow(non_camel_case_types, dead_code)]
        struct $name { __id: $crate::egglog_utils::api::AnySortId }
        #[allow(dead_code)]
        impl $name {
            fn call(&self, $($arg: $crate::egglog_utils::api::Term),*) -> $crate::egglog_utils::api::Term {
                $crate::egglog_utils::api::Term::App {
                    variant: self.__id,
                    args: vec![$($arg),*],
                }
            }
        }
        let __id = $ctx.sort($class, stringify!($name))
            .args([$((stringify!($arg), $sort)),*])
            .build()
            .erase();
        $name { __id }
    }};
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum SortKind {
    Builtin(&'static str),
    User,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct SortClass {
    pub id: SortClassId,
    pub name: String,
    pub kind: SortKind,
    pub variants: Vec<AnySortId>,
    /// If true, this class participates in `(datatype* ...)` mutual recursion.
    pub star: bool,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub sort: SortClassId,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub id: AnySortId,
    pub class: SortClassId,
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct Var {
    pub name: String,
    pub sort: SortClassId,
}

#[derive(Clone, Debug)]
pub enum Literal {
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Unit,
}

#[derive(Clone, Debug)]
pub enum Term {
    Var(Var),
    App {
        variant: AnySortId,
        args: Vec<Term>,
    },
    Lit(Literal),
    /// Egglog primitive/builtin operation: renders as `(name arg1 arg2 ...)`.
    /// Used for egglog built-in ops like `(+ a b)`, `(= ?e X)`, `(union a b)`, etc.
    Prim {
        name: String,
        args: Vec<Term>,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum Directive {
    /// `(rewrite lhs rhs :ruleset rs)`
    Rewrite {
        lhs: String,
        rhs: String,
        ruleset: String,
    },
    /// `(rewrite lhs rhs :when (conds...) :ruleset rs)`
    RewriteWhen {
        lhs: String,
        rhs: String,
        when: String,
        ruleset: String,
    },
    /// `(rewrite lhs rhs :subsume :ruleset rs)`
    RewriteSubsume {
        lhs: String,
        rhs: String,
        ruleset: String,
    },
    /// `(rule (conds...) (actions...) :ruleset rs)`
    Rule {
        conditions: String,
        actions: String,
        ruleset: String,
    },
    /// Arbitrary egglog code, emitted verbatim.
    Raw(String),
}

#[derive(Debug)]
pub struct Program {
    pub(crate) classes: Vec<SortClass>,
    pub(crate) variants: Vec<Variant>,
    pub(crate) rulesets: Vec<String>,
    pub(crate) directives: Vec<Directive>,
}

pub struct VariantBuilder<'a, const N: usize = 0> {
    program: &'a mut Program,
    class: SortClassId,
    name: String,
    fields: Vec<Field>,
}

impl<'a, const N: usize> VariantBuilder<'a, N> {
    pub fn args<const M: usize>(self, args: [(&str, SortClassId); M]) -> VariantBuilder<'a, M> {
        let mut fields = self.fields;
        for (arg_name, arg_sort) in args {
            fields.push(Field {
                name: arg_name.to_string(),
                sort: arg_sort,
            });
        }
        VariantBuilder {
            program: self.program,
            class: self.class,
            name: self.name,
            fields,
        }
    }

    pub fn build(self) -> SortId<N> {
        let mut seen = std::collections::HashSet::new();
        for field in &self.fields {
            if !seen.insert(field.name.as_str()) {
                panic!(
                    "duplicate field name {} in variant {}",
                    field.name, self.name
                );
            }
        }
        if self.fields.len() != N {
            panic!(
                "arity mismatch for {}: expected {}, got {}",
                self.name,
                N,
                self.fields.len()
            );
        }

        let id = SortId::<N>(self.program.variants.len());
        let variant = Variant {
            id: id.erase(),
            class: self.class,
            name: self.name,
            fields: self.fields,
        };

        self.program.variants.push(variant);
        self.program
            .classes
            .get_mut(self.class.0)
            .unwrap_or_else(|| panic!("unknown class id {:?}", self.class))
            .variants
            .push(id.erase());

        id
    }
}

// --- Free functions for constructing terms ---

/// An untyped pattern variable for use in rewrite rules where sort is inferred.
pub fn pat(name: &str) -> Term {
    Term::Var(Var {
        name: name.to_string(),
        sort: SortClassId(usize::MAX),
    })
}

pub fn i64(value: i64) -> Term {
    Term::Lit(Literal::I64(value))
}

pub fn f64(value: f64) -> Term {
    Term::Lit(Literal::F64(value))
}

#[allow(dead_code)]
pub fn bool(value: bool) -> Term {
    Term::Lit(Literal::Bool(value))
}

#[allow(dead_code)]
pub fn str(value: &str) -> Term {
    Term::Lit(Literal::String(value.to_string()))
}

#[allow(dead_code)]
pub fn unit() -> Term {
    Term::Lit(Literal::Unit)
}

/// Generic egglog primitive: renders as `(name arg1 arg2 ...)`.
pub fn prim(name: &str, args: Vec<Term>) -> Term {
    Term::Prim {
        name: name.to_string(),
        args,
    }
}

// --- Egglog built-in i64 primitive operations ---
//
// These are egglog's *primitive* operations on i64 values (the host language's
// arithmetic).  They live in a completely different namespace from the
// user-defined datatype constructors like MAdd, MMul, etc.
//
//   MAdd / MMul / …  – constructors in the Expression *datatype*.  They build
//                       symbolic AST nodes inside the e-graph.
//   iadd / imul / …  – egglog's built-in i64 arithmetic.  They evaluate at
//                       rule-match time to produce concrete i64 results.
//
// Example from base.egg (addition constant-folding):
//
//   (rule ((= ?e (MAdd (MNum a) (MNum b)))   ;  match two symbolic MNum nodes
//          (= ?ans (+ a b)))                  ;  compute the *concrete* sum
//         ((union ?e (MNum ?ans))) …)         ;  merge with the result
//
// The `(+ a b)` here is `iadd(pat("a"), pat("b"))` – it asks egglog to add the
// two matched i64 values at rule-evaluation time.  `MAdd` would instead create
// a *new* symbolic addition node in the e-graph.

pub fn iadd(a: Term, b: Term) -> Term {
    prim("+", vec![a, b])
}
pub fn isub(a: Term, b: Term) -> Term {
    prim("-", vec![a, b])
}
pub fn imul(a: Term, b: Term) -> Term {
    prim("*", vec![a, b])
}
pub fn idiv(a: Term, b: Term) -> Term {
    prim("/", vec![a, b])
}
pub fn imod(a: Term, b: Term) -> Term {
    prim("%", vec![a, b])
}
pub fn imax(a: Term, b: Term) -> Term {
    prim("max", vec![a, b])
}
pub fn imin(a: Term, b: Term) -> Term {
    prim("min", vec![a, b])
}
pub fn iand(a: Term, b: Term) -> Term {
    prim("&", vec![a, b])
}

// --- Egglog comparison primitives ---

pub fn eq(a: Term, b: Term) -> Term {
    prim("=", vec![a, b])
}
pub fn neq(a: Term, b: Term) -> Term {
    prim("!=", vec![a, b])
}
pub fn ilt(a: Term, b: Term) -> Term {
    prim("<", vec![a, b])
}
pub fn igte(a: Term, b: Term) -> Term {
    prim(">=", vec![a, b])
}

// --- Egglog rule action primitives ---

pub fn union(a: Term, b: Term) -> Term {
    prim("union", vec![a, b])
}
pub fn subsume(a: Term) -> Term {
    prim("subsume", vec![a])
}
pub fn set(func_call: Term, value: Term) -> Term {
    prim("set", vec![func_call, value])
}

// --- Program implementation ---

impl Program {
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            variants: Vec::new(),
            rulesets: Vec::new(),
            directives: Vec::new(),
        }
    }

    fn push_class(&mut self, name: &str, kind: SortKind, star: bool) -> SortClassId {
        let id = SortClassId(self.classes.len());
        self.classes.push(SortClass {
            id,
            name: name.to_string(),
            kind,
            variants: Vec::new(),
            star,
        });
        id
    }

    /// Register a user-defined sort class (standalone `(datatype ...)`).
    pub fn class(&mut self, name: &str) -> SortClassId {
        self.push_class(name, SortKind::User, false)
    }

    /// Register a user-defined sort class that participates in `(datatype* ...)`.
    pub fn class_star(&mut self, name: &str) -> SortClassId {
        self.push_class(name, SortKind::User, true)
    }

    /// Register a builtin sort (i64, f64, String, etc.) for use as a field type.
    pub fn builtin(&mut self, egglog_name: &'static str) -> SortClassId {
        self.push_class(egglog_name, SortKind::Builtin(egglog_name), false)
    }

    /// Start building a variant for the given class.
    pub fn sort(&mut self, class: SortClassId, name: &str) -> VariantBuilder<'_> {
        VariantBuilder {
            program: self,
            class,
            name: name.to_string(),
            fields: Vec::new(),
        }
    }

    /// Construct a term using named arguments with validation.
    pub fn apply<const N: usize>(&self, variant: SortId<N>, args: [(&str, Term); N]) -> Term {
        let decl = self
            .variants
            .get(variant.raw())
            .unwrap_or_else(|| panic!("unknown variant id {:?}", variant));

        let mut seen = std::collections::HashSet::new();
        let mut provided = std::collections::HashMap::new();
        for (name, term) in args {
            if !seen.insert(name) {
                panic!("duplicate argument name {} in call to {}", name, decl.name);
            }
            provided.insert(name, term);
        }

        let mut ordered_args = Vec::with_capacity(decl.fields.len());
        for expected_arg in decl.fields.iter() {
            let term = provided
                .remove(expected_arg.name.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "missing argument {} in call to {}",
                        expected_arg.name, decl.name
                    )
                });
            ordered_args.push(term);
        }

        if !provided.is_empty() {
            let extra = provided.keys().cloned().collect::<Vec<_>>().join(", ");
            panic!("unexpected arguments in call to {}: {}", decl.name, extra);
        }

        Term::App {
            variant: variant.erase(),
            args: ordered_args,
        }
    }

    /// Declare a ruleset.
    pub fn ruleset(&mut self, name: &str) {
        self.rulesets.push(name.to_string());
    }

    /// `(rewrite lhs rhs :ruleset rs)`
    pub fn rewrite(&mut self, lhs: Term, rhs: Term, ruleset: &str) {
        self.directives.push(Directive::Rewrite {
            lhs: self.term_to_egglog(&lhs),
            rhs: self.term_to_egglog(&rhs),
            ruleset: ruleset.to_string(),
        });
    }

    /// `(rewrite lhs rhs :when (cond1 cond2 ...) :ruleset rs)`
    pub fn rewrite_when(&mut self, lhs: Term, rhs: Term, conditions: &[Term], ruleset: &str) {
        let when = conditions
            .iter()
            .map(|t| self.term_to_egglog(t))
            .collect::<Vec<_>>()
            .join(" ");
        self.directives.push(Directive::RewriteWhen {
            lhs: self.term_to_egglog(&lhs),
            rhs: self.term_to_egglog(&rhs),
            when,
            ruleset: ruleset.to_string(),
        });
    }

    /// `(rewrite lhs rhs :subsume :ruleset rs)`
    pub fn rewrite_subsume(&mut self, lhs: Term, rhs: Term, ruleset: &str) {
        self.directives.push(Directive::RewriteSubsume {
            lhs: self.term_to_egglog(&lhs),
            rhs: self.term_to_egglog(&rhs),
            ruleset: ruleset.to_string(),
        });
    }

    /// `(rule (cond1 cond2 ...) (action1 action2 ...) :ruleset rs)`
    pub fn egglog_rule(&mut self, conditions: &[Term], actions: &[Term], ruleset: &str) {
        let conds = conditions
            .iter()
            .map(|t| self.term_to_egglog(t))
            .collect::<Vec<_>>()
            .join(" ");
        let acts = actions
            .iter()
            .map(|t| self.term_to_egglog(t))
            .collect::<Vec<_>>()
            .join(" ");
        self.directives.push(Directive::Rule {
            conditions: conds,
            actions: acts,
            ruleset: ruleset.to_string(),
        });
    }

    /// `(function name (Sort1 Sort2 ...) ReturnSort :merge merge)`
    pub fn function_decl(
        &mut self,
        name: &str,
        arg_sorts: &[SortClassId],
        return_sort: SortClassId,
        merge: &str,
    ) {
        let arg_str = arg_sorts
            .iter()
            .map(|s| self.class_name(*s).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let ret_str = self.class_name(return_sort).to_string();
        self.directives.push(Directive::Raw(format!(
            "(function {name} ({arg_str}) {ret_str} :merge {merge})"
        )));
    }

    /// Emit arbitrary egglog code verbatim.
    #[allow(dead_code)]
    pub fn raw(&mut self, code: &str) {
        self.directives.push(Directive::Raw(code.to_string()));
    }

    pub fn class_name(&self, class: SortClassId) -> &str {
        self.classes
            .get(class.0)
            .map(|c| c.name.as_str())
            .unwrap_or("<unknown>")
    }

    pub fn term_to_egglog(&self, term: &Term) -> String {
        match term {
            Term::Var(var) => var.name.to_string(),
            Term::App { variant, args } => {
                let mut out = String::new();
                out.push('(');
                let name = self
                    .variants
                    .get(variant.raw())
                    .map(|v| v.name.as_str())
                    .unwrap_or("<unknown-variant>");
                out.push_str(name);
                for arg in args {
                    out.push(' ');
                    out.push_str(&self.term_to_egglog(arg));
                }
                out.push(')');
                out
            }
            Term::Lit(lit) => match lit {
                Literal::I64(v) => v.to_string(),
                Literal::F64(v) => {
                    let s = v.to_string();
                    if s.contains('.') { s } else { format!("{s}.0") }
                }
                Literal::Bool(b) => if *b { "true" } else { "false" }.to_string(),
                Literal::String(s) => {
                    let mut escaped = String::with_capacity(s.len() + 2);
                    escaped.push('"');
                    for c in s.chars() {
                        match c {
                            '\\' => escaped.push_str("\\\\"),
                            '"' => escaped.push_str("\\\""),
                            '\n' => escaped.push_str("\\n"),
                            '\t' => escaped.push_str("\\t"),
                            c => escaped.push(c),
                        }
                    }
                    escaped.push('"');
                    escaped
                }
                Literal::Unit => "()".to_string(),
            },
            Term::Prim { name, args } => {
                let mut out = String::new();
                out.push('(');
                out.push_str(name);
                for arg in args {
                    out.push(' ');
                    out.push_str(&self.term_to_egglog(arg));
                }
                out.push(')');
                out
            }
        }
    }

    /// Generate the complete egglog source string.
    pub fn to_egglog_string(&self) -> String {
        let mut out = String::new();

        // Emit rulesets
        for rs in &self.rulesets {
            out.push_str(&format!("(ruleset {rs})\n"));
        }
        if !self.rulesets.is_empty() {
            out.push('\n');
        }

        // Collect star classes and standalone classes
        let star_classes: Vec<_> = self
            .classes
            .iter()
            .filter(|c| c.star && matches!(c.kind, SortKind::User))
            .collect();
        let standalone_classes: Vec<_> = self
            .classes
            .iter()
            .filter(|c| !c.star && matches!(c.kind, SortKind::User))
            .collect();

        // Emit (datatype* ...) block
        if !star_classes.is_empty() {
            out.push_str("(datatype*\n");
            for class in &star_classes {
                self.emit_class_body(&mut out, class, true);
            }
            out.push_str(")\n");
        }

        // Emit standalone (datatype ...) blocks
        for class in &standalone_classes {
            if class.variants.is_empty() {
                out.push_str(&format!("(datatype {})\n", class.name));
            } else {
                self.emit_class_body(&mut out, class, false);
            }
        }

        if self
            .classes
            .iter()
            .any(|c| matches!(c.kind, SortKind::User))
        {
            out.push('\n');
        }

        // Emit directives
        for dir in &self.directives {
            match dir {
                Directive::Rewrite { lhs, rhs, ruleset } => {
                    out.push_str(&format!("(rewrite {lhs} {rhs} :ruleset {ruleset})\n"));
                }
                Directive::RewriteWhen {
                    lhs,
                    rhs,
                    when,
                    ruleset,
                } => {
                    out.push_str(&format!(
                        "(rewrite {lhs} {rhs} :when ({when}) :ruleset {ruleset})\n"
                    ));
                }
                Directive::RewriteSubsume { lhs, rhs, ruleset } => {
                    out.push_str(&format!(
                        "(rewrite {lhs} {rhs} :subsume :ruleset {ruleset})\n"
                    ));
                }
                Directive::Rule {
                    conditions,
                    actions,
                    ruleset,
                } => {
                    out.push_str(&format!(
                        "(rule ({conditions}) ({actions}) :ruleset {ruleset})\n"
                    ));
                }
                Directive::Raw(code) => {
                    out.push_str(code);
                    if !code.ends_with('\n') {
                        out.push('\n');
                    }
                }
            }
        }

        out
    }

    fn emit_class_body(&self, out: &mut String, class: &SortClass, inside_star: bool) {
        let indent = if inside_star { "    " } else { "" };
        if inside_star {
            out.push_str(&format!("{indent}({}\n", class.name));
        } else {
            out.push_str(&format!("(datatype {}\n", class.name));
        }
        for variant_id in &class.variants {
            let variant = &self.variants[variant_id.raw()];
            let mut arg_sorts = String::new();
            for field in &variant.fields {
                arg_sorts.push(' ');
                arg_sorts.push_str(self.class_name(field.sort));
            }
            out.push_str(&format!("{indent}    ({}{})\n", variant.name, arg_sorts));
        }
        out.push_str(&format!("{indent})\n"));
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}
