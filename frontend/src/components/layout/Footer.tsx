import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-auto border-t border-[#E4E7EC] dark:border-[#28303E] bg-white dark:bg-[#181D26] py-5 px-4 sm:px-6 transition-colors">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#687385] dark:text-[#8B97A8]">
        <p className="text-center sm:text-left leading-relaxed">
          <strong className="font-medium text-[#202938] dark:text-[#F1F3F7]">Disclaimer:</strong>{' '}
          ClauseGuard is an educational document-analysis tool. It does not provide legal advice or
          replace consultation with a qualified legal professional.
        </p>

        <p className="shrink-0 text-center sm:text-right">
          Documents are processed locally on your device.
        </p>
      </div>
    </footer>
  );
};
