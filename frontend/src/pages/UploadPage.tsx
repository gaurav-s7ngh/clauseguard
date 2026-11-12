import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle } from 'lucide-react';
import { FileDropzone } from '../components/upload/FileDropzone';
import { UploadProgress } from '../components/upload/UploadProgress';
import { CategoryShowcase } from '../components/upload/CategoryShowcase';
import { SampleContractButton } from '../components/upload/SampleContractButton';
import { useDocumentUpload } from '../hooks/useDocumentUpload';
import { useBackendHealth } from '../hooks/useBackendHealth';

export const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const { upload, status, progress, errorMessage, reset } = useDocumentUpload();
  const { isOffline, isMock } = useBackendHealth();
  const [selectedFileName, setSelectedFileName] = useState<string>('');

  const handleFileSelected = async (file: File) => {
    setSelectedFileName(file.name);
    try {
      const res = await upload(file);
      if (res && res.document_id) {
        setTimeout(() => {
          navigate(`/documents/${res.document_id}`);
        }, 800);
      }
    } catch {
      // Handled by upload mutation
    }
  };

  return (
    <div className="flex-1 py-10 sm:py-16 px-4 sm:px-6 max-w-4xl mx-auto w-full space-y-8">
      {/* Header section */}
      <div className="space-y-2 text-center max-w-2xl mx-auto">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#202938] dark:text-[#F1F3F7]">
          Understand your contract before you sign.
        </h1>
        <p className="text-sm text-[#687385] dark:text-[#8B97A8] leading-relaxed">
          Upload a contract to identify important clauses, understand potential risks and review
          suggested changes.
        </p>

        {isOffline && !isMock && (
          <div
            role="alert"
            className="mt-4 flex items-center justify-center gap-2 p-2.5 rounded border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-200 text-xs"
          >
            <AlertCircle className="w-4 h-4 shrink-0 text-amber-600" />
            <span>
              Local analysis service is unreachable. You can still test with a sample contract below.
            </span>
          </div>
        )}
      </div>

      {/* Upload Box */}
      <div className="max-w-xl mx-auto space-y-4">
        <FileDropzone
          onFileSelected={handleFileSelected}
          disabled={status === 'uploading' || status === 'analyzing'}
        />

        <UploadProgress
          status={status}
          progress={progress}
          fileName={selectedFileName}
          errorMessage={errorMessage}
          onRetry={reset}
        />

        <div className="text-center pt-1">
          <SampleContractButton />
        </div>
      </div>

      {/* What ClauseGuard checks */}
      <div className="max-w-3xl mx-auto pt-6">
        <CategoryShowcase />
      </div>
    </div>
  );
};
