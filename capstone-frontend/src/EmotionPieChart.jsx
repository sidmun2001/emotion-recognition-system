import React from "react";
import { PieChart } from "react-minimal-pie-chart";


const EmotionPieChart = () => {

    const data = [
        { emotion: "Happy", accuracy: 35 },
        { emotion: "Sad", accuracy: 20 },
        { emotion: "Angry", accuracy: 10 },
        { emotion: "Surprised", accuracy: 15 },
        { emotion: "Neutral", accuracy: 10 },
        { emotion: "Fearful", accuracy: 5 },
        { emotion: "Disgusted", accuracy: 5 },
      ];
  // Generate color codes dynamically
  const colors = ["#E38627", "#C13C37", "#6A2135", "#3D9970", "#0074D9", "#FFDC00", "#FF851B"];

  // Format data for Pie Chart
  const chartData = data.map((item, index) => ({
    title: item.emotion,
    value: item.accuracy,
    color: colors[index % colors.length], // Cycle through colors
  }));

  return (
    <div className="border-2 flex justify-center items-center w-full h-full p-4 m-2">
        <div className=" border-2 ">
        <h2 className="text-lg font-semibold mb-4">Emotion Recognition Accuracy</h2>
        <PieChart
            data={chartData}
            label={({ dataEntry }) => `${dataEntry.title} (${dataEntry.value}%)`}
            labelStyle={{ fontSize: "5px", fill: "#fff" }}
            radius={42}
            labelPosition={112}
        />
        </div>
    </div>
  );
};

export default EmotionPieChart;
