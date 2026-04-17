import { Component, type ReactNode } from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Navbar from "./components/Navbar";
import Analytics from "./components/Analytics";
import BuyMeCoffee from "./components/BuyMeCoffee";
import UploadPage from "./pages/UploadPage";
import ProcessingPage from "./pages/ProcessingPage";
import DashboardPage from "./pages/DashboardPage";
import TableDetailPage from "./pages/TableDetailPage";
import SettingsPage from "./pages/SettingsPage";

class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-surface-950 flex items-center justify-center px-6">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-surface-100 mb-2">Something went wrong</h1>
            <p className="text-surface-300 mb-4">An unexpected error occurred.</p>
            <button
              onClick={() => { this.setState({ hasError: false }); window.location.href = "/"; }}
              className="px-5 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 transition-colors cursor-pointer"
            >
              Return to home
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Analytics />
        <div className="min-h-screen bg-surface-950">
          <Navbar />
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/processing/:jobId" element={<ProcessingPage />} />
            <Route path="/results/:jobId" element={<DashboardPage />} />
            <Route path="/results/:jobId/table/:tableName" element={<TableDetailPage />} />
            <Route path="*" element={
              <div className="min-h-[calc(100vh-56px)] flex items-center justify-center px-6">
                <div className="text-center">
                  <h1 className="text-4xl font-bold text-surface-100 mb-2">404</h1>
                  <p className="text-surface-300 mb-6">Page not found</p>
                  <Link to="/" className="px-5 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 transition-colors">
                    Back to home
                  </Link>
                </div>
              </div>
            } />
          </Routes>
          <BuyMeCoffee />
        </div>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
