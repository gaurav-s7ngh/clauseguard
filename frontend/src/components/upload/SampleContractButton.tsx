import React from 'react';
import { useNavigate } from 'react-router-dom';

interface SampleContractButtonProps {
  className?: string;
}

export const SampleContractButton: React.FC<SampleContractButtonProps> = ({ className }) => {
  const navigate = useNavigate();

  const handleLoadSample = () => {
    navigate('/documents/doc_mock_sample_master');
  };

  return (
    <button
      onClick={handleLoadSample}
      type="button"
      className={
        className ||
        'inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#315C8A] dark:text-[#5D8EBF] hover:text-[#202938] dark:hover:text-white border border-[#D0D5DD] dark:border-[#384252] rounded bg-white dark:bg-[#181D26] hover:bg-[#F1F3F6] dark:hover:bg-[#202633] transition-colors'
      }
    >
      <span>Try a sample contract</span>
      <span aria-hidden="true">&rarr;</span>
    </button>
  );
};
