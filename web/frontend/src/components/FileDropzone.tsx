import { useCallback, useState } from "react";
import { Upload, FileJson, FileSpreadsheet, FileText, FileCode } from "lucide-react";
import clsx from "clsx";

const EXT_ICONS: Record<string, typeof FileJson> = {
  ".json": FileJson,
  ".csv": FileText,
  ".xml": FileCode,
  ".xlsx": FileSpreadsheet,
  ".xls": FileSpreadsheet,
};

const ACCEPT = ".json,.csv,.xml,.xlsx,.xls";

interface Props {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function FileDropzone({ onFileSelect, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
        if (!ACCEPT.split(",").includes(ext)) return;
        setSelected(file);
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        setSelected(file);
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const ext = selected ? selected.name.slice(selected.name.lastIndexOf(".")).toLowerCase() : "";
  const Icon = EXT_ICONS[ext] || Upload;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Drop zone: drag and drop a data file here, or click to browse"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={clsx(
        "relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer",
        "flex flex-col items-center justify-center gap-4 p-12",
        "group hover:border-brand-500/60 hover:bg-brand-500/5",
        dragging
          ? "border-brand-400 bg-brand-500/10 scale-[1.02]"
          : "border-surface-300/20 bg-surface-900/50",
        disabled && "opacity-50 pointer-events-none"
      )}
    >
      <input
        type="file"
        accept={ACCEPT}
        onChange={handleChange}
        className="absolute inset-0 opacity-0 cursor-pointer"
        disabled={disabled}
        aria-label="Upload data file"
      />

      <div
        className={clsx(
          "w-16 h-16 rounded-xl flex items-center justify-center transition-all",
          "bg-brand-500/10 group-hover:bg-brand-500/20",
          selected ? "text-brand-400" : "text-surface-300"
        )}
      >
        <Icon size={28} />
      </div>

      {selected ? (
        <div className="text-center">
          <p className="text-surface-100 font-medium text-lg">{selected.name}</p>
          <p className="text-surface-300 text-sm mt-1">
            {(selected.size / 1024).toFixed(1)} KB &middot;{" "}
            {ext.replace(".", "").toUpperCase()}
          </p>
        </div>
      ) : (
        <div className="text-center">
          <p className="text-surface-200 font-medium text-lg">
            Drop your data file here
          </p>
          <p className="text-surface-300 text-sm mt-1">
            JSON, CSV, XML, or Excel &middot; up to 100 MB
          </p>
        </div>
      )}

      <div className="flex gap-2 mt-2">
        {Object.entries(EXT_ICONS).map(([extension, ExtIcon]) => (
          <span
            key={extension}
            className="px-2.5 py-1 rounded-md bg-surface-800 text-surface-300 text-xs font-mono flex items-center gap-1.5"
          >
            <ExtIcon size={12} />
            {extension}
          </span>
        ))}
      </div>
    </div>
  );
}
