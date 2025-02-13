import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import axios from 'axios';





function Header() 
{
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const dropdownRef = useRef(null);



  // Close dropdown if clicking outside
  useEffect(() => {
    function handleClickOutside(event) 
    {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) 
      {
        setDropdownVisible(false);
      }
    }


    document.addEventListener('mousedown', handleClickOutside);
    return () => 
    {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);



  // Function to delete images
  const deleteImages = async () => 
  {
    try 
    {
      const response = await axios.post('http://localhost:8080/delete-images');
      if (response.status === 200) 
      {
        alert('All images deleted successfully!');
      } 
      
      else 
      {
        alert('Failed to delete images.');
      }
    } 
    
    catch (error) 
    {
      console.error('Error deleting images:', error);
      alert('Error deleting images.');
    }
  };



  return (
    <div className="header">
      {/* DROPDOWN MENU */}
      <div className="dropdown" ref={dropdownRef}>
        <button 
          className="dropdown-btn" 
          onClick={() => setDropdownVisible(!dropdownVisible)}
        >
          ☰
        </button>
        {dropdownVisible && (
          <ul className="dropdown-content">
            <li onClick={deleteImages}>Delete</li>
          </ul>
        )}
      </div>


      {/* HEADER TEXT */}
      <h1>Emotion Recognition System</h1>
    </div>
  );
}



export default Header;