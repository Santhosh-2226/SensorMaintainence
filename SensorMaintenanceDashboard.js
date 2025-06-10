import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertTriangle, CheckCircle, AlertOctagon } from 'lucide-react';
import BarChart from 'recharts/lib/chart/BarChart';
import Bar from 'recharts/lib/cartesian/Bar';
import ScatterChart from 'recharts/lib/chart/ScatterChart';
import Scatter from 'recharts/lib/cartesian/Scatter';

const SensorMaintenanceDashboard = () => {
  const [sensorData, setSensorData] = useState([]);
  const [maintenanceWarnings, setMaintenanceWarnings] = useState([]);
  const [vehicleStatus, setVehicleStatus] = useState('operational');
  const [featureImportance, setFeatureImportance] = useState([]);
  const [rocData, setRocData] = useState([]);

  useEffect(() => {
    const generateSensorData = () => {
      const newData = [];
      const warningsList = [];
      let isVehicleStopped = false;
      const importanceData = [
        { feature: 'Temperature', importance: 0.4 },
        { feature: 'Humidity', importance: 0.3 },
        { feature: 'Wind Speed', importance: 0.2 },
        { feature: 'Maintenance Score', importance: 0.1 },
      ];

      const rocCurve = [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.1, tpr: 0.7 },
        { fpr: 0.3, tpr: 0.9 },
        { fpr: 0.6, tpr: 1.0 },
      ];

      for (let i = 0; i < 50; i++) {
        const temperature = 20 + Math.random() * 20;
        const humidity = 40 + Math.random() * 40;
        const windSpeed = Math.random() * 15;

        const maintenanceScore =
          (temperature > 35 ? 0.7 : 0) +
          (temperature < 0 ? 0.7 : 0) +
          (humidity > 80 ? 0.5 : 0) +
          (windSpeed > 10 ? 0.6 : 0);

        const maintenanceNeeded = maintenanceScore > 0.6;

        const dataPoint = {
          timestamp: i,
          temperature,
          humidity,
          windSpeed,
          maintenanceScore,
          anomaly: maintenanceScore > 0.8 ? 'Anomalous' : 'Normal',
        };

        newData.push(dataPoint);

        if (maintenanceNeeded) {
          warningsList.push({
            timestamp: i,
            severity: maintenanceScore > 0.8 ? 'critical' : 'warning',
            message:
              maintenanceScore > 0.8
                ? 'CRITICAL: Immediate Maintenance Required'
                : 'Warning: Potential Maintenance Needed',
          });

          if (maintenanceScore > 0.8 && !isVehicleStopped) {
            isVehicleStopped = true;
            setVehicleStatus('stopped');
          }
        }
      }

      setSensorData(newData);
      setMaintenanceWarnings(warningsList);
      setFeatureImportance(importanceData);
      setRocData(rocCurve);
    };

    generateSensorData();
  }, []);

  const getStatusIcon = () => {
    switch (vehicleStatus) {
      case 'operational':
        return <CheckCircle color="green" size={48} />;
      case 'stopped':
        return <AlertOctagon color="red" size={48} />;
      default:
        return <AlertTriangle color="orange" size={48} />;
    }
  };

  return (
    <div className="p-4 bg-gray-100 min-h-screen">
      <div className="bg-white shadow-md rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-4">Sensor Maintenance Dashboard</h1>

        <div className="flex items-center mb-4">
          <span className="mr-2">Vehicle Status:</span>
          <div className="flex items-center">
            {getStatusIcon()}
            <span
              className={`ml-2 font-semibold ${
                vehicleStatus === 'stopped' ? 'text-red-600' : 'text-green-600'
              }`}
            >
              {vehicleStatus.toUpperCase()}
            </span>
          </div>
        </div>

        <div className="mb-6 h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <XAxis dataKey="timestamp" />
              <YAxis />
              <CartesianGrid />
              <Scatter data={sensorData} fill="blue" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="mb-6 h-64">
          <BarChart width={600} height={300} data={featureImportance}>
            <Bar dataKey="importance" fill="lightgreen" />
          </BarChart>
        </div>
      </div>
    </div>
  );
};

export default SensorMaintenanceDashboard;
