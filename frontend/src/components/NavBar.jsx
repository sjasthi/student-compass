import React from 'react';
import { Link, useLocation } from 'react-router-dom';

/**
 * NavBar provides top navigation with links to Chat, Admin, and Test.
 * Highlights the active route based on the current location.
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
        <li>
          <Link
            to="/test"
            className={
              isActive('/test')
                ? 'font-semibold text-blue-300'
                : 'hover:text-blue-300'
            }
          >
            Test
          </Link>
        </li>
      </ul>
    </nav>
  );
}
