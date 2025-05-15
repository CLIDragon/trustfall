use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use trustfall_core::ir::TransparentValue;

// This file consists of lines directly copied from cargo-semver-checks.
// In the future, I expect tracing to be part of the API exposed by trustfall,
// which will be enabled in cargo-semver-checks itself.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RequiredSemverUpdate {
    #[serde(alias = "minor")]
    Minor,
    #[serde(alias = "major")]
    Major,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LintLevel {
    /// If this lint occurs, do nothing.
    #[serde(alias = "allow")]
    Allow,
    /// If this lint occurs, print a warning.
    #[serde(alias = "warn")]
    Warn,
    /// If this lint occurs, raise an error.
    #[serde(alias = "deny")]
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemverQuery {
    pub id: String,

    pub(crate) human_readable_name: String,

    pub description: String,

    pub required_update: RequiredSemverUpdate,

    /// The default lint level for when this lint occurs.
    pub lint_level: LintLevel,

    #[serde(default)]
    pub reference: Option<String>,

    #[serde(default)]
    pub reference_link: Option<String>,

    pub(crate) query: String,

    #[serde(default)]
    pub(crate) arguments: BTreeMap<String, TransparentValue>,

    /// The top-level error describing the semver violation that was detected.
    /// Even if multiple instances of this semver issue are found, this error
    /// message is displayed only at most once.
    pub(crate) error_message: String,

    /// Optional template that can be combined with each query output to produce
    /// a human-readable description of the specific semver violation that was discovered.
    #[serde(default)]
    pub(crate) per_result_error_template: Option<String>,

    /// Optional data to create witness code for query output.  See the [`Witness`] struct for
    /// more information.
    #[serde(default)]
    pub witness: Option<Witness>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Witness {
    pub hint_template: String,

    #[serde(default)]
    pub witness_template: Option<String>,

    #[serde(default)]
    pub witness_query: Option<WitnessQuery>,
}

/// A [`trustfall`] query, for [`Witness`] generation, containing the query
/// string itself and a mapping of argument names to value types which are
/// provided to the query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessQuery {
    /// The string containing the Trustfall query.
    pub query: String,

    /// The mapping of argument names to values provided to the query.
    ///
    /// These can be inherited from a previous query ([`InheritedValue::Inherited`]) or
    /// specified as [`InheritedValue::Constant`]s.
    #[serde(default)]
    pub arguments: BTreeMap<String, InheritedValue>,
}

/// Represents either a value inherited from a previous query, or a
/// provided constant value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum InheritedValue {
    /// Inherit the value from the previous output whose name is the given `String`.
    Inherited { inherit: String },
    /// Provide the constant value specified here.
    Constant(TransparentValue),
}
