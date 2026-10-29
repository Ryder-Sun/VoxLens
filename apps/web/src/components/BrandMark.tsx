import { Link } from "@tanstack/react-router";

export function BrandMark() {
  return (
    <Link to="/" className="group inline-flex items-center">
      <img
        src="/brand/voxlens-logo-full.svg"
        alt="VoxLens"
        height={40}
        className="h-10 w-auto transition-transform duration-500 group-hover:scale-[1.03]"
      />
    </Link>
  );
}
