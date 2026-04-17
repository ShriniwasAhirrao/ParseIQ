import { Coffee } from "lucide-react";

/**
 * Floating "Buy me a coffee" button.
 * Set VITE_BMC_USERNAME to your buymeacoffee.com handle to enable it.
 */
export default function BuyMeCoffee() {
  const username = import.meta.env.VITE_BMC_USERNAME as string | undefined;
  if (!username) return null;

  return (
    <a
      href={`https://www.buymeacoffee.com/${username}`}
      target="_blank"
      rel="noopener noreferrer"
      className="fixed bottom-6 right-6 z-40 group flex items-center gap-2 pl-3 pr-4 py-2.5 rounded-full bg-gradient-to-br from-amber-400 to-amber-500 text-amber-950 text-sm font-semibold shadow-lg shadow-amber-500/30 hover:shadow-xl hover:shadow-amber-500/40 hover:scale-105 transition-all duration-200"
      aria-label="Buy me a coffee"
    >
      <Coffee size={16} className="group-hover:animate-bounce" />
      <span>Buy me a coffee</span>
    </a>
  );
}
