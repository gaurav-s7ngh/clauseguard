import React, { useRef, useState, type DragEvent, type ChangeEvent } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { validateContractFile } from '../../hooks/useDocumentUpload';
import { cn } from '../../lib/utils';

interface FileDropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export const FileDropzone: React.FC<FileDropzoneProps> = ({ onFileSelected, disabled }) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    setIsDragActive(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    const result = validateContractFile(file);
    if (!result.valid) {
      setValidationError(result.error || 'Invalid file.');
      return;
    }
    setValidationError(null);
    onFileSelected(file);
  };

  const triggerPicker = () => {
    if (disabled) return;
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full space-y-3">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerPicker}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            triggerPicker();
          }
        }}
        tabIndex={disabled ? -1 : 0}
        role="button"
        aria-label="Upload legal contract: Drag and drop PDF, DOCX or TXT file or click to browse"
        aria-disabled={disabled}
        className={cn(
          'border border-dashed rounded-lg p-8 sm:p-10 text-center transition-colors cursor-pointer bg-white dark:bg-[#181D26] outline-none focus-visible:ring-2 focus-visible:ring-[#315C8A]',
          isDragActive
            ? 'border-[#315C8A] bg-[#EDF3FA] dark:bg-[#1E2C3D]'
            : 'border-[#D0D5DD] dark:border-[#384252] hover:border-[#315C8A] dark:hover:border-[#5D8EBF]',
          disabled && 'opacity-60 cursor-not-allowed'
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          onChange={handleFileInputChange}
          disabled={disabled}
          className="hidden"
          aria-hidden="true"
        />

        <div className="max-w-md mx-auto flex flex-col items-center space-y-3">
          <div className="w-10 h-10 rounded-full bg-[#F1F3F6] dark:bg-[#202633] text-[#315C8A] dark:text-[#5D8EBF] flex items-center justify-center">
            <Upload className="w-5 h-5" />
          </div>

          <div className="space-y-1">
            <p className="text-sm font-medium text-[#202938] dark:text-[#F1F3F7]">
              Choose a contract or drag and drop
            </p>
            <p className="text-xs text-[#687385] dark:text-[#8B97A8]">
              PDF, DOCX, or TXT (up to 20 MB)
            </p>
          </div>

          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              triggerPicker();
            }}
            disabled={disabled}
            className="mt-2 inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded text-xs font-medium bg-[#315C8A] hover:bg-[#264A70] text-white transition-colors"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Select file</span>
          </button>
        </div>
      </div>

      {validationError && (
        <div
          role="alert"
          className="flex items-center gap-2 p-3 rounded border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 text-red-800 dark:text-red-300 text-xs"
        >
          <AlertCircle className="w-4 h-4 shrink-0 text-red-600" />
          <span>{validationError}</span>
        </div>
      )}
    </div>
  );
};
