import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home  from './pages/Home.jsx';
import Admin from './pages/Admin.jsx';
import Test  from './pages/Test.jsx';
import NavBar from './components/NavBar.jsx';

/**
 * Root application component.
 * Sets up client-side routing:
 *   /       → Chat (Home)
 *   /admin  → Admin upload management
 *   /test   → RAG evaluation test runner
 */
function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/"      element={<Home  />} />
        <Route path="/admin" element={<Admin />} />
        <Route path="/test"  element={<Test  />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
