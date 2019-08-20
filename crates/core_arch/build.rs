fn main() -> io::Result<()> {
    println!("cargo:rustc-cfg=core_arch_docs");
}
