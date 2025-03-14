import React from "react";
import { PieChart } from "react-minimal-pie-chart";
import './App.css';



const EmotionPieChart = ({ emotionData }) => 
{
  if (!emotionData) 
  {
    return <p>No Data Available</p>;
  }



  // Fixed color mapping for each emotion
  const emotionColors = 
  {
    "Anger": "#FF0000",     // Red
    "Disgust": "#00CED1",   // Brown
    "Fear": "#800080",      // Purple
    "Happiness": "#FFD700",     // Gold
    "Sadness": "#1E90FF",       // Blue
    "Surprise": "#FF69B4",  // Pink
    "Neutral": "#A9A9A9"    // Gray
  };



  // Format data for Pie Chart with fixed colors
  const chartData = Object.entries(emotionData).map
  (([emotion, value]) => 
    (
      {
        title: emotion,
        value,
        color: emotionColors[emotion] || "#000000", // Default to black if emotion is missing
      }
    )
  );



  return (
    <div className="emotion-chart-container">
      

      <div className="legend">
        <div className="legend-items">
          {chartData.map((entry, index) => (
            <div key={index} className="legend-item">
              <span className="legend-color" style={{ backgroundColor: entry.color }}></span>
              <span>{entry.title}: {entry.value}%</span>
            </div>
          ))}
        </div>
      </div>


      {/* Pie Chart */}
      <h3 style={{ marginBottom: "-150px" }}></h3>
      <PieChart
        data={chartData}
        radius={30}
        labelPosition={110}
      />
    </div>
  );
}




export default EmotionPieChart;