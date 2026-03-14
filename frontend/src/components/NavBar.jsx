import React from 'react';
import { Link, useLocation } from 'react-router-dom';

/**
 * NavBar component provides a simple top navigation bar with links
 * to the chat home page and the admin upload page. It highlights
 * the active route based on the current location. This component
 * relies on react-router-dom's Link and useLocation hooks to
 * perform client-side navigation without full page reloads.
 */
export default function NavBar() {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-gray-800 text-white px-4 py-3 shadow-md">
      <ul className="flex space-x-4">
        <li>
          <Link
            to="/"
            className={
              isActive('/')
                ? 'font-semibold text-blue-300'
                : 'hover:text-blue-300'
            }
          >
            Chat
          </Link>
        </li>
        <li>
          <Link
            to="/admin"
            className={
              isActive('/admin')
                ? 'font-semibold text-blue-300'
                : 'hover:text-blue-300'
            }
          >
            Admin
          </Link>
        </li>
      </ul>
    </nav>
  );
}