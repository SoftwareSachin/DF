import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-900">
          Deep Fake Detection System
        </h1>
        <p className="text-gray-600 mt-1">
          AI-powered analysis for authentic media verification
        </p>
      </div>
    </header>
  );
};

export default Header;