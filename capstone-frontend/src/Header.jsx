import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Header = () => {
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const navigate = useNavigate();

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest('.dropdown')) {
        setDropdownVisible(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, []);

  const handleDropdownAction = async (action) => {
    switch (action) {
      case 'restart':
        navigate('/');
        break;
      case 'delete':
        try {
          const response = await axios.post('http://localhost:8080/delete-images');
          if (response.status === 200) {
            alert('All images deleted successfully.');
          } else {
            alert('Failed to delete images.');
          }
        } catch (error) {
          console.error('Error deleting images:', error);
        }
        break;
      case 'exit':
        alert('Exit action may not work due to browser restrictions.');
        break;
      default:
        console.log('Unknown action:', action);
    }
  };

  return (
    <div className="header">
      {/* Dropdown Menu */}
      <div className="dropdown">
        <button
          className="dropdown-toggle"
          onClick={(e) => {
            e.stopPropagation(); // Prevent triggering outside click event
            setDropdownVisible(!dropdownVisible);
          }}
        >
          ☰ {/* Hamburger icon */}
        </button>
        <ul className={`dropdown-menu ${dropdownVisible ? 'visible' : ''}`}>
          <li onClick={() => handleDropdownAction('restart')}>Restart</li>
          <li onClick={() => handleDropdownAction('delete')}>Delete</li>
          <li onClick={() => handleDropdownAction('exit')}>Exit</li>
        </ul>
      </div>
      {/* Title */}
      <h1 className="header-title">Emotion Recognition System</h1>
    </div>
  );
};

export default Header;