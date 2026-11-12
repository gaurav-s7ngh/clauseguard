import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import type { DocumentStatus } from '../types/analysis';

const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024; // 20 MB
const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt'];
const ALLOWED_MIME_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'text/markdown'
];

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export function validateContractFile(file: File): ValidationResult {
  if (!file) {
    return { valid: false, error: 'Please select a valid contract file.' };
  }

  // Size check
  if (file.size > MAX_FILE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return {
      valid: false,
      error: `File size (${sizeMb} MB) exceeds the maximum 20 MB limit.`
    };
  }

  // Extension check
  const lowerName = file.name.toLowerCase();
  const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
  const hasValidMime = ALLOWED_MIME_TYPES.includes(file.type);

  if (!hasValidExt && !hasValidMime) {
    return {
      valid: false,
      error: 'Unsupported file format. Please upload a PDF (.pdf), Microsoft Word (.docx), or plain text (.txt) document.'
    };
  }

  return { valid: true };
}

export function useDocumentUpload() {
  const [status, setStatus] = useState<DocumentStatus>('idle');
  const [progress, setProgress] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadedDocId, setUploadedDocId] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: async (file: File) => {
      setErrorMessage(null);
      const validation = validateContractFile(file);
      if (!validation.valid) {
        throw new Error(validation.error);
      }

      setStatus('uploading');
      setProgress(25);

      // Upload file
      const uploadRes = await apiClient.uploadDocument(file);
      setProgress(50);
      setUploadedDocId(uploadRes.document_id);

      setStatus('analyzing');
      setProgress(75);

      // Trigger analysis pipeline
      await apiClient.startAnalysis(uploadRes.document_id);
      setProgress(100);
      setStatus('completed');

      return uploadRes;
    },
    onError: (err: Error) => {
      setStatus('failed');
      setErrorMessage(err.message || 'An error occurred during upload or analysis.');
    }
  });

  const reset = () => {
    setStatus('idle');
    setProgress(0);
    setErrorMessage(null);
    setUploadedDocId(null);
    mutation.reset();
  };

  return {
    upload: mutation.mutateAsync,
    isUploading: mutation.isPending,
    status,
    progress,
    errorMessage,
    uploadedDocId,
    reset
  };
}

