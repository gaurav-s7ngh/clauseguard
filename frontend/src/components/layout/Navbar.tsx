import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FileText, Moon, Sun } from 'lucide-react';
import { useBackendHealth } from '../../hooks/useBackendHealth';
import { useTheme } from '../../hooks/useTheme';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const { isMock, isHealthy } = useBackendHealth();
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="w-full border-b border-[#E4E7EC] dark:border-[#28303E] bg-white dark:bg-[#181D26] transition-colors">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        {/* Brand */}
        <Link
          to="/"
          className="flex items-center gap-2.5 text-[#202938] dark:text-white hover:opacity-90 transition-opacity"
        >
          <div className="w-7 h-7 rounded bg-[#315C8A] text-white flex items-center justify-center font-bold text-sm">
            <FileText className="w-4 h-4" />
          </div>
          <span className="font-semibold text-base tracking-tight">ClauseGuard</span>
        </Link>

        {/* Navigation & Utilities */}
        <div className="flex items-center gap-6 text-sm">
          <nav className="flex items-center gap-4">
            <Link
              to="/"
              className={`font-medium transition-colors ${
                location.pathname === '/'
                  ? 'text-[#315C8A] dark:text-[#5D8EBF]'
                  : 'text-[#687385] dark:text-[#8B97A8] hover:text-[#202938] dark:hover:text-white'
              }`}
            >
              Analyzer
            </Link>

            <Link
              to="/history"
              className={`font-medium transition-colors ${
                location.pathname === '/history'
                  ? 'text-[#315C8A] dark:text-[#5D8EBF]'
                  : 'text-[#687385] dark:text-[#8B97A8] hover:text-[#202938] dark:hover:text-white'
              }`}
            >
              History
            </Link>
          </nav>

          <div className="h-4 w-px bg-[#E4E7EC] dark:bg-[#28303E]" />

          {/* Connection status dot */}
          <div
            className="flex items-center gap-2 text-xs text-[#687385] dark:text-[#8B97A8]"
            title={
              isMock
                ? 'Running in sample offline mode'
                : isHealthy
                  ? 'Connected to local analysis service'
                  : 'Analysis service offline'
            }
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isMock ? 'bg-amber-500' : isHealthy ? 'bg-emerald-500' : 'bg-red-500'
              }`}
            />
            <span className="hidden sm:inline">
              {isMock ? 'Sample mode' : isHealthy ? 'Connected' : 'Offline'}
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            type="button"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="p-1.5 rounded text-[#687385] dark:text-[#8B97A8] hover:text-[#202938] dark:hover:text-white hover:bg-[#F1F3F6] dark:hover:bg-[#202633] transition-colors"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </header>
  );
};
