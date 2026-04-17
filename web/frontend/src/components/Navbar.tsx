import { Settings, Zap } from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";

export default function Navbar() {
  const navigate = useNavigate();

  return (
    <nav className="sticky top-0 z-50 border-b border-surface-300/10 bg-surface-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2.5 hover:opacity-80 transition-opacity cursor-pointer"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-lg shadow-brand-500/30">
            <Zap size={16} className="text-white" fill="currentColor" />
          </div>
          <span className="text-surface-100 font-bold text-lg tracking-tight font-[family-name:var(--font-display)]">
            ParseIQ
          </span>
          <span className="text-[10px] text-surface-300 bg-surface-800 px-1.5 py-0.5 rounded font-mono">
            v0.0.6
          </span>
        </button>

        <div className="flex items-center gap-4 text-xs text-surface-300">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              "flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer " +
              (isActive
                ? "bg-brand-500/10 text-brand-300"
                : "hover:bg-surface-800/60 hover:text-surface-100")
            }
          >
            <Settings size={13} />
            <span>Settings</span>
          </NavLink>
          <span className="hidden md:flex items-center gap-4">
            <a
              href="https://pypi.org/project/parseiq/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-surface-100 transition-colors"
            >
              PyPI
            </a>
            <a
              href="https://github.com/ShriniwasAhirrao/ParseIQ"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-surface-100 transition-colors"
            >
              GitHub
            </a>
          </span>
        </div>
      </div>
    </nav>
  );
}
