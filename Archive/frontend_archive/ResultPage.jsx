import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Header from './Header';




const ResultPage = () => 
{
  // Use the useLocation hook to retrieve state
  const location = useLocation();
  const navigate = useNavigate();
  console.log('location:', location);


  const uploadedImageUrl = 'http://127.0.0.1:8080/uploads/captured_image.jpg';
  const preprocessedImageUrl = 'http://127.0.0.1:8080/preprocessed/preprocessed_captured_image.jpg';
  const emotion = 'not implemented yet';
  console.log('uploadedImageUrl:', uploadedImageUrl);
  console.log('preprocessedImageUrl:', preprocessedImageUrl);
  console.log('emotion:', emotion);

  const handleBackToHome = () => 
  {
    navigate('/')
  };
  


  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Facial Emotion Recognition Result</h1>
      <h2>Uploaded Image:</h2>
      <img src={uploadedImageUrl} alt="Uploaded image" style={{ width: '300px', height: '300px', objectFit: 'cover' }} />
      <h2>Uploaded Image:</h2>
      <img src={preprocessedImageUrl} alt="Preprocessed image" style={{ width: '300px', height: '300px', objectFit: 'cover' }} />

      <h2>Recognized Emotion: {emotion}</h2>

      <button onClick={handleBackToHome} className="back-to-home-btn">
        Back to Home
      </button>
    </div>
  );
};




export default ResultPage; 