import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home.jsx';
import Admin from './pages/Admin.jsx';
import NavBar from './components/NavBar.jsx';

/**
 * Root application component. Simply renders the Home page.
 */
/**
 * Root application component. It sets up client-side routing using
 * react-router-dom. The navigation bar is displayed at the top of
 * every page, and Routes determine which page to render based on
 * the current path. The home page remains at '/', while the admin
 * upload form is available at '/admin'.
 */
function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/admin" element={<Admin />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;