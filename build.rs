fn main() {
    println!("cargo:rustc-link-lib=dylib=sqlite3");
    println!("cargo:rustc-link-search=native={}",std::env::current_dir().unwrap().to_str().unwrap());
}
