import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { validateContractFile } from '../hooks/useDocumentUpload';
import { FileDropzone } from '../components/upload/FileDropzone';

describe('Upload Validation & FileDropzone Component', () => {
  it('accepts valid PDF, DOCX, and TXT files under 20 MB', () => {
    const validPdf = new File(['%PDF-1.4 mock content'], 'contract.pdf', {
      type: 'application/pdf'
    });
    const validDocx = new File(['mock binary content'], 'agreement.docx', {
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    });
    const validTxt = new File(['Section 1. Terms...'], 'terms.txt', {
      type: 'text/plain'
    });

    expect(validateContractFile(validPdf).valid).toBe(true);
    expect(validateContractFile(validDocx).valid).toBe(true);
    expect(validateContractFile(validTxt).valid).toBe(true);
  });

  it('rejects files exceeding 20 MB limit with a clear error', () => {
    // 21 MB file
    const largeFile = new File(['a'.repeat(100)], 'huge_contract.pdf', {
      type: 'application/pdf'
    });
    Object.defineProperty(largeFile, 'size', { value: 21 * 1024 * 1024 });

    const result = validateContractFile(largeFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('exceeds the maximum 20 MB limit');
  });

  it('rejects unsupported file formats with an explanatory error', () => {
    const invalidImage = new File(['fake image bytes'], 'screenshot.png', {
      type: 'image/png'
    });
    const result = validateContractFile(invalidImage);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('Unsupported file format');
  });

  it('renders FileDropzone with accessible role and instructions', () => {
    const handleSelected = vi.fn();
    render(<FileDropzone onFileSelected={handleSelected} />);

    expect(screen.getByRole('button', { name: /upload legal contract/i })).toBeInTheDocument();
    expect(screen.getByText(/Choose a contract or drag and drop/i)).toBeInTheDocument();
    expect(screen.getByText(/PDF, DOCX, or TXT/i)).toBeInTheDocument();
  });

  it('displays validation alert when user selects an invalid file', () => {
    const handleSelected = vi.fn();
    const { container } = render(<FileDropzone onFileSelected={handleSelected} />);

    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const invalidFile = new File(['test'], 'malicious.exe', { type: 'application/x-msdownload' });

    fireEvent.change(input, { target: { files: [invalidFile] } });

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/Unsupported file format/i)).toBeInTheDocument();
    expect(handleSelected).not.toHaveBeenCalled();
  });

  it('calls onFileSelected when a valid contract is uploaded', () => {
    const handleSelected = vi.fn();
    const { container } = render(<FileDropzone onFileSelected={handleSelected} />);

    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const validFile = new File(['Valid terms of service text'], 'tos.txt', { type: 'text/plain' });

    fireEvent.change(input, { target: { files: [validFile] } });

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(handleSelected).toHaveBeenCalledTimes(1);
    expect(handleSelected).toHaveBeenCalledWith(validFile);
  });
});

