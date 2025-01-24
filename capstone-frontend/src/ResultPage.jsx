import React from 'react';
import { useLocation } from 'react-router-dom';
import Header from './Header';



const ResultPage = () => {

  // Use the useLocation hook to retrieve state
  const location = useLocation();
  //const { uploadedImageUrl, preprocessedImageUrl, emotion } = location.state || {};

    const uploadedImageUrl = 'http://127.0.0.1:8080/uploads/captured_image.jpg';
    const preprocessedImageUrl = 'http://127.0.0.1:8080/preprocessed/preprocessed_captured_image.jpg';
    const emotion = location.state.emotion;
    console.log('uploadedImageUrl:', uploadedImageUrl);
    console.log('preprocessedImageUrl:', preprocessedImageUrl);
    console.log('emotion:', emotion);
  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Facial Emotion Recognition Result</h1>
      <h2>Uploaded Image:</h2>
      <img src={uploadedImageUrl} alt="Uploaded image" style={{ width: '300px', height: '300px', objectFit: 'cover' }} />
      <h2>Uploaded Image:</h2>
      <img src={preprocessedImageUrl} alt="Preprocessed image" style={{ width: '300px', height: '300px', objectFit: 'cover' }} />

      <h2>Recognized Emotion: {emotion}</h2>
    </div>
  );
};



export default ResultPage; 