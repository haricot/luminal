use std::collections::HashMap;

// ========== Core Types ==========

/// A sort class (type) — either a builtin like `i64` or a user-defined datatype like `Expr`.
#[derive(Clone, Copy, Debug)]
pub struct SortClass {
    pub name: &'static str,
}

impl SortClass {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

/// All primitive (non-parameterized) builtin sorts in egglog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinSort {
    I64,
    F64,
    Bool,
    String,
    BigInt,
    BigRat,
    Unit,
}

impl BuiltinSort {
    pub const ALL: [BuiltinSort; 7] = [
        BuiltinSort::I64,
        BuiltinSort::F64,
        BuiltinSort::Bool,
        BuiltinSort::String,
        BuiltinSort::BigInt,
        BuiltinSort::BigRat,
        BuiltinSort::Unit,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BuiltinSort::I64 => "i64",
            BuiltinSort::F64 => "f64",
            BuiltinSort::Bool => "bool",
            BuiltinSort::String => "String",
            BuiltinSort::BigInt => "BigInt",
            BuiltinSort::BigRat => "BigRat",
            BuiltinSort::Unit => "Unit",
        }
    }

    pub fn sort_class(self) -> SortClass {
        SortClass::new(self.name())
    }
}

/// Parameterized (container) builtin sorts in egglog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Presort {
    Vec,
    Set,
    Map,
    MultiSet,
}

impl Presort {
    pub fn name(self) -> &'static str {
        match self {
            Presort::Vec => "Vec",
            Presort::Set => "Set",
            Presort::Map => "Map",
            Presort::MultiSet => "MultiSet",
        }
    }

    pub fn arity(self) -> usize {
        match self {
            Presort::Vec | Presort::Set | Presort::MultiSet => 1,
            Presort::Map => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub sort: String,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub class: String,
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct Var {
    pub name: String,
    pub sort: String,
}

/// Literal values for egglog's primitive builtin sorts.
#[derive(Clone, Debug)]
pub enum Literal {
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Unit,
}

/// Term — the core AST node for building egglog expressions.
#[derive(Clone, Debug)]
pub enum Term {
    Var(Var),
    App { variant: String, args: Vec<Term> },
    Lit(Literal),
}

/// An action in a rule body.
#[derive(Clone, Debug)]
pub enum Action {
    Union(Term, Term),
    Set(Term, Term),
    Subsume(Term),
    Delete(Term),
}

/// An egglog rule. `(rewrite lhs rhs)` is just sugar for
/// `(rule ((= ?v lhs)) ((union ?v rhs)))`, so this single type
/// represents both. Use [`rewrite()`] for the sugar, [`rule()`] for general rules.
#[derive(Clone, Debug)]
pub struct Rule {
    pub name: Option<String>,
    pub facts: Vec<Term>,
    pub actions: Vec<Action>,
    pub ruleset: Option<String>,
}

/// Fresh variable name used internally by [`rewrite()`] desugaring.
const RW_VAR: &str = "?__rw";

impl Rule {
    pub fn new() -> Self {
        Self {
            name: None,
            facts: vec![],
            actions: vec![],
            ruleset: None,
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn fact(mut self, fact: Term) -> Self {
        self.facts.push(fact);
        self
    }

    pub fn facts(mut self, facts: Vec<Term>) -> Self {
        self.facts = facts;
        self
    }

    pub fn action(mut self, action: Action) -> Self {
        self.actions.push(action);
        self
    }

    pub fn actions(mut self, actions: Vec<Action>) -> Self {
        self.actions = actions;
        self
    }

    pub fn ruleset(mut self, name: &str) -> Self {
        self.ruleset = Some(name.to_string());
        self
    }

    /// Extend the fact list with additional conditions.
    pub fn when(mut self, conditions: Vec<Term>) -> Self {
        self.facts.extend(conditions);
        self
    }

    /// Sugar for `.action(Action::Set(func_app, value))`.
    pub fn set(self, func_app: Term, value: Term) -> Self {
        self.action(Action::Set(func_app, value))
    }

    /// Sugar for `.action(Action::Union(a, b))`.
    pub fn union(self, a: Term, b: Term) -> Self {
        self.action(Action::Union(a, b))
    }

    /// Sugar for `.fact(eq(var, term))` — binds a pattern variable via equality.
    pub fn r#let(self, var: Term, term: Term) -> Self {
        self.fact(eq(var, term))
    }

    /// Convert this rule to its egglog string representation.
    pub fn to_egglog_string(&self) -> String {
        rule_to_egglog(self)
    }
}

/// A function declaration in egglog.
#[derive(Clone, Debug)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<String>,
    pub ret: String,
    pub merge: Option<String>,
}

// ========== Free-standing Sort Definition ==========

/// A sort variant definition that has not yet been registered into a `Program`.
#[derive(Clone, Debug)]
pub struct SortDef {
    pub class: String,
    pub name: String,
    pub fields: Vec<Field>,
}

impl SortDef {
    /// Construct a `Term::App` from this sort definition with named arguments.
    pub fn call<const N: usize>(&self, args: [(&str, Term); N]) -> Term {
        assert_eq!(
            N,
            self.fields.len(),
            "sort `{}` expects {} args, got {}",
            self.name,
            self.fields.len(),
            N
        );

        let mut provided: HashMap<&str, Term> = args.into_iter().collect();

        let mut ordered = Vec::with_capacity(N);
        for field in &self.fields {
            let term = provided.remove(field.name.as_str()).unwrap_or_else(|| {
                panic!(
                    "missing argument `{}` in call to `{}`",
                    field.name, self.name
                )
            });
            ordered.push(term);
        }

        if !provided.is_empty() {
            let extra: Vec<_> = provided.keys().copied().collect();
            panic!(
                "unexpected arguments in call to `{}`: {}",
                self.name,
                extra.join(", ")
            );
        }

        Term::App {
            variant: self.name.clone(),
            args: ordered,
        }
    }
}

// ========== Free-standing Builders ==========

/// Create a sort variant definition.
pub fn sort(class: &SortClass, name: &str, args: &[(&str, &SortClass)]) -> SortDef {
    let mut seen = std::collections::HashSet::new();
    let mut fields = Vec::with_capacity(args.len());
    for (arg_name, arg_sort) in args {
        if !seen.insert(*arg_name) {
            panic!("duplicate field name {} in variant {}", arg_name, name);
        }
        fields.push(Field {
            name: arg_name.to_string(),
            sort: arg_sort.name.to_string(),
        });
    }
    SortDef {
        class: class.name.to_string(),
        name: name.to_string(),
        fields,
    }
}

/// Create a rule in rewrite form. `(rewrite lhs rhs)` is syntactic sugar for
/// `(rule ((= ?v lhs)) ((union ?v rhs)))`.
pub fn rewrite(name: &str, lhs: Term, rhs: Term) -> Rule {
    Rule {
        name: Some(name.to_string()),
        facts: vec![eq(v(RW_VAR), lhs)],
        actions: vec![Action::Union(v(RW_VAR), rhs)],
        ruleset: None,
    }
}

/// Create a general rule.
pub fn rule() -> Rule {
    Rule::new()
}

/// Create a pattern variable.
pub fn var(name: &str, _sort: &SortClass) -> Term {
    Term::Var(Var {
        name: name.to_string(),
        sort: String::new(),
    })
}

/// Create an untyped pattern variable (sort is not tracked).
pub fn v(name: &str) -> Term {
    Term::Var(Var {
        name: name.to_string(),
        sort: String::new(),
    })
}

pub fn i64(value: i64) -> Term {
    Term::Lit(Literal::I64(value))
}

pub fn f64(value: f64) -> Term {
    Term::Lit(Literal::F64(value))
}

pub fn bool(value: bool) -> Term {
    Term::Lit(Literal::Bool(value))
}

pub fn str(value: &str) -> Term {
    Term::Lit(Literal::String(value.to_string()))
}

pub fn unit() -> Term {
    Term::Lit(Literal::Unit)
}

/// Create a function/builtin definition (for term construction only, not registered as a sort).
pub fn func(name: &str, arg_names: &[&str]) -> SortDef {
    SortDef {
        class: String::new(),
        name: name.to_string(),
        fields: arg_names
            .iter()
            .map(|n| Field {
                name: n.to_string(),
                sort: String::new(),
            })
            .collect(),
    }
}

/// Sort/function application — builds a term from a `SortDef` and positional arguments.
pub fn app(sort: &SortDef, args: Vec<Term>) -> Term {
    assert_eq!(
        args.len(),
        sort.fields.len(),
        "`{}` expects {} args, got {}",
        sort.name,
        sort.fields.len(),
        args.len()
    );
    Term::App {
        variant: sort.name.clone(),
        args,
    }
}

/// Egglog equality: `(= a b)`
pub fn eq(a: Term, b: Term) -> Term {
    Term::App {
        variant: "=".to_string(),
        args: vec![a, b],
    }
}

/// Egglog inequality: `(!= a b)`
pub fn neq(a: Term, b: Term) -> Term {
    Term::App {
        variant: "!=".to_string(),
        args: vec![a, b],
    }
}

// ========== Code Generation ==========

pub fn term_to_egglog(term: &Term) -> String {
    match term {
        Term::Var(var) => var.name.to_string(),
        Term::App { variant, args } => {
            let mut out = String::new();
            out.push('(');
            out.push_str(variant);
            for arg in args {
                out.push(' ');
                out.push_str(&term_to_egglog(arg));
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
    }
}

fn action_to_egglog(action: &Action) -> String {
    match action {
        Action::Union(a, b) => format!("(union {} {})", term_to_egglog(a), term_to_egglog(b)),
        Action::Set(func_app, value) => {
            format!(
                "(set {} {})",
                term_to_egglog(func_app),
                term_to_egglog(value)
            )
        }
        Action::Subsume(t) => format!("(subsume {})", term_to_egglog(t)),
        Action::Delete(t) => format!("(delete {})", term_to_egglog(t)),
    }
}

fn rule_to_egglog(rule: &Rule) -> String {
    let facts: Vec<_> = rule.facts.iter().map(term_to_egglog).collect();
    let actions: Vec<_> = rule.actions.iter().map(action_to_egglog).collect();
    let mut out = format!("(rule ({}) ({})", facts.join(" "), actions.join(" "));
    if let Some(rs) = &rule.ruleset {
        out.push_str(&format!(" :ruleset {}", rs));
    }
    out.push(')');
    out
}

fn function_to_egglog(func: &FunctionDef) -> String {
    let args = func.args.join(" ");
    let mut out = format!("(function {} ({}) {}", func.name, args, func.ret);
    if let Some(merge) = &func.merge {
        out.push_str(&format!(" :merge {}", merge));
    }
    out.push(')');
    out
}

// ========== Program ==========

/// Items that can be added to a Program, preserving insertion order.
#[derive(Clone, Debug)]
pub enum ProgramItem {
    Rule(Rule),
    Function(FunctionDef),
}

#[derive(Clone, Debug)]
pub enum ProgramSortKind {
    Builtin,
    BuiltinInst { presort: Presort, args: Vec<String> },
    User,
}

#[derive(Clone, Debug)]
pub struct ProgramSortClass {
    pub name: String,
    pub kind: ProgramSortKind,
    pub variants: Vec<String>,
}

#[derive(Debug)]
pub struct Program {
    pub rulesets: Vec<String>,
    pub classes: Vec<ProgramSortClass>,
    pub variants: Vec<Variant>,
    pub items: Vec<ProgramItem>,
    pub mutual_recursive: bool,
}

impl Program {
    pub fn new() -> Self {
        let mut program = Self {
            rulesets: Vec::new(),
            classes: Vec::new(),
            variants: Vec::new(),
            items: Vec::new(),
            mutual_recursive: false,
        };

        for b in BuiltinSort::ALL {
            program.classes.push(ProgramSortClass {
                name: b.name().to_string(),
                kind: ProgramSortKind::Builtin,
                variants: Vec::new(),
            });
        }

        program
    }

    pub fn add_ruleset(&mut self, name: &str) {
        self.rulesets.push(name.to_string());
    }

    pub fn add_class(&mut self, class: &SortClass) {
        if self.classes.iter().any(|c| c.name == class.name) {
            panic!("class `{}` is already registered", class.name);
        }
        self.classes.push(ProgramSortClass {
            name: class.name.to_string(),
            kind: ProgramSortKind::User,
            variants: Vec::new(),
        });
    }

    pub fn add_sort(&mut self, def: &SortDef) {
        let class = self
            .classes
            .iter_mut()
            .find(|c| c.name == def.class)
            .unwrap_or_else(|| panic!("unknown class `{}`", def.class));
        class.variants.push(def.name.clone());

        self.variants.push(Variant {
            class: def.class.clone(),
            name: def.name.clone(),
            fields: def.fields.clone(),
        });
    }

    pub fn add_rule(&mut self, rule: Rule) {
        self.items.push(ProgramItem::Rule(rule));
    }

    pub fn add_function(&mut self, func: FunctionDef) {
        self.items.push(ProgramItem::Function(func));
    }

    pub fn to_egglog_string(&self) -> String {
        let mut out = String::new();

        // Emit rulesets
        for rs in &self.rulesets {
            out.push_str(&format!("(ruleset {})\n", rs));
        }
        if !self.rulesets.is_empty() {
            out.push('\n');
        }

        // Emit parameterized sort instantiations
        for class in &self.classes {
            if let ProgramSortKind::BuiltinInst { presort, args } = &class.kind {
                out.push_str(&format!(
                    "(sort {} ({} {}))\n",
                    class.name,
                    presort.name(),
                    args.join(" ")
                ));
            }
        }

        // Emit user-defined datatypes
        let user_classes: Vec<_> = self
            .classes
            .iter()
            .filter(|c| matches!(c.kind, ProgramSortKind::User))
            .collect();

        if !user_classes.is_empty() {
            if self.mutual_recursive {
                // Emit as (datatype* ...)
                out.push_str("(datatype*\n");
                for class in &user_classes {
                    out.push_str(&format!("    ({}\n", class.name));
                    for variant_name in &class.variants {
                        let variant = self
                            .variants
                            .iter()
                            .find(|v| v.name == *variant_name)
                            .unwrap();
                        let mut arg_sorts = String::new();
                        for field in &variant.fields {
                            arg_sorts.push(' ');
                            arg_sorts.push_str(&field.sort);
                        }
                        out.push_str(&format!("        ({}{})\n", variant.name, arg_sorts));
                    }
                    out.push_str("    )\n");
                }
                out.push_str(")\n");
            } else {
                // Emit individual (datatype ...) blocks
                for class in &user_classes {
                    if class.variants.is_empty() {
                        out.push_str(&format!("(datatype {})\n", class.name));
                        continue;
                    }

                    out.push_str(&format!("(datatype {}\n", class.name));
                    for variant_name in &class.variants {
                        let variant = self
                            .variants
                            .iter()
                            .find(|v| v.name == *variant_name)
                            .unwrap();
                        let mut arg_sorts = String::new();
                        for field in &variant.fields {
                            arg_sorts.push(' ');
                            arg_sorts.push_str(&field.sort);
                        }
                        out.push_str(&format!("  ({}{})\n", variant.name, arg_sorts));
                    }
                    out.push_str(")\n");
                }
            }
        }
        out.push('\n');

        // Emit items in insertion order
        for item in &self.items {
            match item {
                ProgramItem::Rule(r) => {
                    out.push_str(&rule_to_egglog(r));
                    out.push('\n');
                }
                ProgramItem::Function(f) => {
                    out.push_str(&function_to_egglog(f));
                    out.push('\n');
                }
            }
        }

        out
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}
