//! Forward the native link directives of the `kernels` crate to this crate's
//! own dependents.
//!
//! `cargo:rustc-link-*` instructions only apply to the crate whose build
//! script emitted them.  The CUDA/C++ archives are built by the `kernels`
//! build script, so a binary or example that depends on `attention-rs`
//! directly would otherwise miss them on its link line and fail with
//! "undefined reference to std::__throw_system_error" and friends.
//!
//! The `kernels` crate declares `links = "kernels"`, so cargo hands its
//! metadata to this build script through `DEP_KERNELS_*`.  Re-emitting the
//! instructions here propagates them one level further.

use std::env;

/// The `DEP_KERNELS_*` keys the `kernels` build script exports.
const LINK_KEYS: &[&str] = &[
    "lib_pagedattention",
    "lib_flashmla",
    "lib_cudart",
    "lib_cublas",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-env-changed=DEP_KERNELS_LIB_SEARCH"
    );
    for key in LINK_KEYS {
        println!("cargo:rerun-if-env-changed=DEP_KERNELS_{}", key.to_uppercase());
    }

    // Only the CUDA build produces native archives; the Metal build has no
    // static libraries to link.
    let search = match env::var("DEP_KERNELS_LIB_SEARCH") {
        Ok(v) if !v.is_empty() => v,
        _ => return,
    };
    println!("cargo:rustc-link-search={search}");

    for key in LINK_KEYS {
        if let Ok(lib) = env::var(format!("DEP_KERNELS_{}", key.to_uppercase())) {
            if !lib.is_empty() {
                println!("cargo:rustc-link-lib={lib}");
            }
        }
    }

    // The archives are C++ and use the libstdc++ ABI.  Consumers that do not
    // otherwise pull in a C++ dependency still need those symbols.
    if !cfg!(target_env = "msvc") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
