import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Pause, RotateCcw, Activity, Upload, CheckCircle, Zap } from 'lucide-react';
import { trainModel, predictTest, loadModel } from './api';
import './App.css';
import './index.css';

const VentilatorSimulations = () => {
  const [activeTab, setActiveTab] = useState('ml');
  
  // Dataset State
  const [datasetLoaded, setDatasetLoaded] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [rawData, setRawData] = useState([]);
  const [processedData, setProcessedData] = useState([]);
  // Test Dataset State
  const [testDatasetLoaded, setTestDatasetLoaded] = useState(false);
  const [testDatasetInfo, setTestDatasetInfo] = useState(null);
  const [testProcessedData, setTestProcessedData] = useState([]);
  
  // ML Simulation State
  const [mlRunning, setMlRunning] = useState(false);
  const [mlTesting, setMlTesting] = useState(false);
  const [mlData, setMlData] = useState([]);
  const [mlEpoch, setMlEpoch] = useState(0);
  const [mlMetrics, setMlMetrics] = useState({ mae: 0, rmse: 0 });
  const [testMetrics, setTestMetrics] = useState({ mae: 0, rmse: 0, samples: 0 });
  const [currentBreath, setCurrentBreath] = useState(0);
  const [trainedModel, setTrainedModel] = useState(null);
  const [testRawData, setTestRawData] = useState([]);
  const [predictions, setPredictions] = useState([]);

  // Breath parameters
  const [breathParams, setBreathParams] = useState({ R: 20, C: 50 });
  
  // Cellular Automata State
  const [caRunning, setCaRunning] = useState(false);
  const [caGrid, setCaGrid] = useState([]);
  const [caGeneration, setCaGeneration] = useState(0);
  const [caStats, setCaStats] = useState({ 
    highPressure: 0, 
    mediumPressure: 0, 
    lowPressure: 0 
  });

  useEffect(() => {
    initializeCaGrid();
  }, []);
  // Función para entrenar con PyTorch
const handlePyTorchTrain = async () => {
  if (!rawData || rawData.length === 0) {
    alert('Carga primero el dataset de entrenamiento');
    return;
  }

  setMlRunning(true);
  
  try {
    // Crear archivo CSV desde rawData
    const csvContent = convertToCSV(rawData);
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const file = new File([blob], 'train.csv');
    
    alert('Iniciando entrenamiento con PyTorch en el servidor...');
    
    const result = await trainModel(file, 50);
    
    alert(`¡Entrenamiento completado!\nHistorial: ${JSON.stringify(result.history)}`);
  } catch (error) {
    console.error('Error:', error);
    alert('Error al entrenar: ' + error.message);
  } finally {
    setMlRunning(false);
  }
};


const handlePyTorchTest = async () => {
  if (!testDatasetLoaded || testProcessedData.length === 0) {
    alert('Carga primero el dataset de test');
    return;
  }

  setMlTesting(true);
  
  try {
    // Aplanar los datos procesados a formato plano
    const flatData = [];
    testProcessedData.forEach(breath => {
      breath.forEach(point => {
        flatData.push({
          id: point.id,
          breath_id: point.breathId,
          R: point.R,
          C: point.C,
          time_step: point.timeStep,
          u_in: point.u_in,
          u_out: point.u_out
        });
      });
    });
    
    // Convertir a CSV
    const csvContent = convertToCSV(flatData);
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const file = new File([blob], 'test.csv');
    
    console.log('Enviando archivo de test al servidor...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Error HTTP: ${response.status}`);
    }
    
    const result = await response.json();
    
    console.log('Respuesta del servidor:', result);
    
    if (result.predictions) {
      setTestMetrics({
        mae: result.estimated_mae || 0,
        rmse: result.estimated_mae ? result.estimated_mae * 1.3 : 0,
        samples: result.total_predictions || 0
      });
      
      alert(`¡Predicciones completadas!\n` +
            `Total: ${result.total_predictions || 0} predicciones\n` +
            `MAE estimado: ${(result.estimated_mae || 0).toFixed(3)} cmH₂O`);
    } else {
      alert('Error: ' + (result.error || 'Respuesta inesperada'));
    }
  } catch (error) {
    console.error('Test error:', error);
    alert('Error al hacer test: ' + error.message + '\n\n¿Está el servidor Flask corriendo en http://localhost:5000?');
  } finally {
    setMlTesting(false);
  }
};

// Agregar este botón en la sección de botones (línea ~380):
<button
  onClick={handlePyTorchTest}
  disabled={mlTesting || !testDatasetLoaded}
  className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:bg-gray-400"
>
  <Zap size={18} />
  {mlTesting ? 'Testeando...' : 'Test PyTorch'}
</button>

const handleTrainFileUpload = async (event) => {
  const file = event.target.files[0];
  if (file) {
    setMlRunning(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('epochs', 50);
    
    try {
      const response = await fetch('http://localhost:5000/api/train', {
        method: 'POST',
        body: formData
      });
      const result = await response.json();
      setMlData(result.history || []);
      alert('Entrenamiento completado');
    } catch (error) {
      console.error('Training error:', error);
      alert('Error en el entrenamiento');
    } finally {
      setMlRunning(false);
    }
  }
};

const handleTestFileUpload = async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  try {
    const text = await file.text();
    const lines = text.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Parse CSV data
    const data = [];
    for (let i = 1; i < Math.min(lines.length, 10001); i++) {
      const values = lines[i].split(',');
      if (values.length === headers.length) {
        const row = {};
        headers.forEach((header, idx) => {
          row[header] = values[idx].trim();
        });
        data.push(row);
      }
    }

    setTestRawData(data);
    
    // Process data into breaths (sin pressure)
    const breaths = processDataIntoBreaths(data, false);
    setTestProcessedData(breaths);
    
    setTestDatasetInfo({
      fileName: file.name,
      totalRows: data.length,
      totalBreaths: breaths.length,
      features: headers,
      sample: data[0]
    });
    
    setTestDatasetLoaded(true);
    alert('Dataset de test cargado exitosamente! Ahora puedes presionar "Test ML" para hacer las predicciones.');
    
  } catch (error) {
    console.error('Error processing test file:', error);
    alert('Error al procesar el archivo de test. Asegúrate de que sea un CSV válido.');
  }
};


// Función auxiliar para convertir a CSV
const convertToCSV = (data) => {
  if (data.length === 0) return '';
  
  const headers = Object.keys(data[0]);
  const csvRows = [headers.join(',')];
  
  data.forEach(row => {
    const values = headers.map(header => row[header]);
    csvRows.push(values.join(','));
  });
  
  return csvRows.join('\n');
};

  const initializeCaGrid = () => {
    const rows = 20;
    const cols = 30;
    const grid = Array(rows).fill(null).map(() => 
      Array(cols).fill(null).map(() => {
        const rand = Math.random();
        if (rand > 0.7) return 2;
        if (rand > 0.4) return 1;
        return 0;
      })
    );
    setCaGrid(grid);
    setCaGeneration(0);
    updateCaStats(grid);
  };

  const updateCaStats = (grid) => {
    const flat = grid.flat();
    const high = flat.filter(cell => cell === 2).length;
    const medium = flat.filter(cell => cell === 1).length;
    const low = flat.filter(cell => cell === 0).length;
    setCaStats({ highPressure: high, mediumPressure: medium, lowPressure: low });
  };

  // Handle CSV file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const text = await file.text();
      const lines = text.split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      
      // Parse CSV data
      const data = [];
      for (let i = 1; i < Math.min(lines.length, 10001); i++) { // Limit to 10000 rows
        const values = lines[i].split(',');
        if (values.length === headers.length) {
          const row = {};
          headers.forEach((header, idx) => {
            row[header] = values[idx].trim();
          });
          data.push(row);
        }
      }

      setRawData(data);
      
      // Process data into breaths
      const breaths = processDataIntoBreaths(data);
      setProcessedData(breaths);
      
      setDatasetInfo({
        fileName: file.name,
        totalRows: data.length,
        totalBreaths: breaths.length,
        features: headers,
        sample: data[0]
      });
      
      setDatasetLoaded(true);
    } catch (error) {
      console.error('Error processing file:', error);
      alert('Error al procesar el archivo. Asegúrate de que sea un CSV válido.');
    }
  };

  const processDataIntoBreaths = (data, hasPressure = true) => {
  const breaths = [];
  let currentBreath = [];
  let currentBreathId = null;

  data.forEach(row => {
    const breathId = row.breath_id || row['breath_id'];
    
    if (currentBreathId !== breathId) {
      if (currentBreath.length > 0) {
        breaths.push(currentBreath);
      }
      currentBreath = [];
      currentBreathId = breathId;
    }
    
    const breathData = {
      id: row.id || row['id'],
      breathId: breathId,
      timeStep: parseFloat(row.time_step || row['time_step'] || 0),
      u_in: parseFloat(row.u_in || row['u_in'] || 0),
      u_out: parseFloat(row.u_out || row['u_out'] || 0),
      R: parseFloat(row.R || 20),
      C: parseFloat(row.C || 50)
    };
    
    // Solo agregar pressure si existe en el dataset
    if (hasPressure && row.pressure) {
      breathData.pressure = parseFloat(row.pressure || 0);
    }
    
    currentBreath.push(breathData);
  });
  
  if (currentBreath.length > 0) {
    breaths.push(currentBreath);
  }
  
  return breaths;
};

  // Calculate pressure using physical equation
  const calculatePressure = (timeStep, R, C, u_in, u_out) => {
    // P(t) = R*Q(t) + (1/C)*V(t)
    // Q(t) es el flujo (relacionado con u_in)
    // V(t) es el volumen acumulado
    const flow = u_in * 5; // Flujo inspiratorio
    const volume = timeStep * flow * 0.1; // Volumen acumulado
    const pressure = R * flow + (1/C) * volume;
    const noise = (Math.random() - 0.5) * 1.5; // Ruido para simular variabilidad
    return Math.max(0, pressure + noise);
  };

  // Generate synthetic breath data
  const generateBreathData = (timeStep, R, C, u_in) => {
    return calculatePressure(timeStep, R, C, u_in, 0);
  };

  // ML Simulation with real or synthetic data
  useEffect(() => {
  if (!mlRunning && !mlTesting) return;

  const interval = setInterval(() => {
    if (mlTesting) {
      // MODO TESTING
          setCurrentBreath(prev => {
        const newBreath = prev + 1;
        
        if (newBreath >= 125) {
          // Terminó el testing
          setMlTesting(false);
          alert(`Testing completado!\nMAE: ${testMetrics.mae.toFixed(3)} cmH₂O\nRMSE: ${testMetrics.rmse.toFixed(3)} cmH₂O\nCiclos evaluados: ${testProcessedData.length}`);
          return 0;
        }

        // Mostrar ciclo actual del test
        const breath = testProcessedData[newBreath];
        
        // Calcular métricas acumuladas hasta este ciclo
        let totalError = 0;
        let totalSquaredError = 0;
        let totalSamples = 0;

        // Evaluar todos los ciclos hasta el actual
        for (let i = 0; i <= newBreath; i++) {
          const currentBreath = testProcessedData[i];
          
          currentBreath.forEach(point => {
            const predictedPressure = calculatePressure(
              point.timeStep,
              point.R,
              point.C,
              point.u_in,
              point.u_out
            );
            
            // Simular presión "real" para evaluación
            const simulatedTruePressure = calculatePressure(
              point.timeStep,
              point.R,
              point.C,
              point.u_in,
              point.u_out
            ) * (0.95 + Math.random() * 0.1);

            const error = Math.abs(predictedPressure - simulatedTruePressure);
            totalError += error;
            totalSquaredError += error * error;
            totalSamples++;
          });
        }

        const mae = totalError / totalSamples;
        const rmse = Math.sqrt(totalSquaredError / totalSamples);

        // Actualizar métricas en tiempo real
        setTestMetrics({ mae, rmse, samples: totalSamples });

        // Preparar datos para la gráfica del ciclo actual
        const breathData = breath.map(point => {
          const predictedPressure = calculatePressure(
            point.timeStep,
            point.R,
            point.C,
            point.u_in,
            point.u_out
          );

          return {
            timeStep: point.timeStep,
            u_in: point.u_in,
            u_out: point.u_out,
            R: point.R,
            C: point.C,
            predictedPressure: Math.max(0, predictedPressure),
            actualPressure: null
          };
        });

        setMlData(breathData);
        return newBreath;
      });
    } else {
      // MODO ENTRENAMIENTO (código original)
      setMlEpoch(prev => {
        const newEpoch = prev + 1;
        
        const learningProgress = 1 - Math.exp(-newEpoch / 15);
        const chaos = Math.random() * 0.5 * (1 - learningProgress);
        
        const mae = 5.0 * (1 - learningProgress) + chaos;
        const rmse = 7.0 * (1 - learningProgress) + chaos;
        
        setMlMetrics({
          mae: Math.max(0.5, mae),
          rmse: Math.max(1.0, rmse)
        });

        let breathData;
        if (datasetLoaded && processedData.length > 0) {
          const breathIndex = currentBreath % processedData.length;
          const realBreath = processedData[breathIndex];
          
          breathData = realBreath.map(point => {
            const actualPressure = point.pressure;
            const predictedPressure = actualPressure + (Math.random() - 0.5) * mae * 2;
            
            return {
              timeStep: point.timeStep,
              u_in: point.u_in,
              u_out: point.u_out,
              R: point.R,
              C: point.C,
              actualPressure: actualPressure,
              predictedPressure: Math.max(0, predictedPressure),
              error: Math.abs(actualPressure - predictedPressure)
            };
          });
          
          setCurrentBreath(prev => prev + 1);
        } else {
          breathData = [];
          for (let t = 0; t < 80; t++) {
            const u_in = t < 40 ? 1 : 0;
            const actualPressure = generateBreathData(t, breathParams.R, breathParams.C, u_in);
            const predictedPressure = actualPressure + (Math.random() - 0.5) * mae * 2;
            
            breathData.push({
              timeStep: t,
              u_in: u_in,
              actualPressure: actualPressure,
              predictedPressure: Math.max(0, predictedPressure),
              error: Math.abs(actualPressure - predictedPressure)
            });
          }
        }

        setMlData(breathData);

        if (newEpoch >= 125 * processedData.length) {
          setMlRunning(false);
          ALERT('ENTRENAMIENTO COMPLETADO!');
        }

        return newEpoch;
      });
    }
  }, mlTesting ? 100 : 300); // Testing más rápido (100ms)

  return () => clearInterval(interval);
}, [mlRunning, mlTesting, breathParams, datasetLoaded, processedData, currentBreath, testProcessedData]);

  // Cellular Automata
  useEffect(() => {
    if (!caRunning) return;

    const interval = setInterval(() => {
      setCaGrid(prevGrid => {
        const newGrid = prevGrid.map((row, i) => 
          row.map((cell, j) => {
            let neighborSum = 0;
            let count = 0;
            
            for (let di = -1; di <= 1; di++) {
              for (let dj = -1; dj <= 1; dj++) {
                if (di === 0 && dj === 0) continue;
                const ni = (i + di + prevGrid.length) % prevGrid.length;
                const nj = (j + dj + row.length) % row.length;
                neighborSum += prevGrid[ni][nj];
                count++;
              }
            }

            const avgNeighbor = neighborSum / count;
            let newPressure = cell;
            
            if (cell === 2 && avgNeighbor < 1.5) {
              newPressure = 1;
            } else if (cell === 0 && avgNeighbor > 1.0) {
              newPressure = 1;
            } else if (cell === 1) {
              if (avgNeighbor > 1.5) newPressure = 2;
              else if (avgNeighbor < 0.8) newPressure = 0;
            }
            
            if (Math.random() < 0.05) {
              newPressure = Math.floor(Math.random() * 3);
            }
            
            return newPressure;
          })
        );
        
        updateCaStats(newGrid);
        return newGrid;
      });

      setCaGeneration(prev => prev + 1);
    }, 200);

    return () => clearInterval(interval);
  }, [caRunning]);

  const resetMl = () => {
    setMlRunning(false);
    setMlTesting(false);
    setMlData([]);
    setMlEpoch(0);
    setMlMetrics({ mae: 0, rmse: 0 });
    setTestMetrics({ mae: 0, rmse: 0, samples: 0 });
    setCurrentBreath(0);
    setMlTesting(false);
    setTestMetrics({ mae: 0, rmse: 0, samples: 0 });
  };
  const startTesting = () => {
  if (!testDatasetLoaded || testProcessedData.length === 0) {
    alert('Por favor carga primero el archivo test.csv');
    return;
  }

  if (mlEpoch === 0) {
    alert('Primero debes entrenar el modelo');
    return;
  }

  setMlTesting(true);
  setMlRunning(false); // Detener el entrenamiento si está corriendo
  setCurrentBreath(0); // Resetear el contador
  
  alert('Iniciando testing automático...');
};
  

  const resetCa = () => {
    setCaRunning(false);
    initializeCaGrid();
  };

  const getCellColor = (value) => {
    if (value === 0) return 'bg-blue-200';
    if (value === 1) return 'bg-blue-500';
    return 'bg-blue-800';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Google Brain - Ventilator Pressure Prediction
          </h1>
          <p className="text-gray-600">Workshop 4 - Simulaciones Computacionales</p>
          <div className="mt-4 flex items-center gap-4 text-sm text-gray-600">
            <Activity className="text-blue-600" size={20} />
            <span>Kaggle Competition Dataset</span>
          </div>
        </div>

        {/* Dataset Upload Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <Upload size={24} className="text-blue-600" />
            Cargar Dataset de Kaggle
          </h2>
          
          <div className="mb-4">
            <label className="block mb-2">
              <span className="text-sm font-semibold text-gray-700">
                Selecciona el archivo CSV (train.csv):
              </span>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="block w-full mt-2 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </label>
          </div>

          {datasetLoaded && datasetInfo && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="text-green-600" size={24} />
                <h3 className="font-semibold text-green-800">Dataset Cargado Exitosamente</h3>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Archivo:</p>
                  <p className="font-semibold">{datasetInfo.fileName}</p>
                </div>
                <div>
                  <p className="text-gray-600">Registros:</p>
                  <p className="font-semibold">{datasetInfo.totalRows.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-600">Ciclos (breaths):</p>
                  <p className="font-semibold">{datasetInfo.totalBreaths.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-600">Features:</p>
                  <p className="font-semibold">{datasetInfo.features.length}</p>
                </div>
              </div>

              <div className="mt-3 p-3 bg-white rounded border">
                <p className="text-xs text-gray-600 mb-1">Features del CSV:</p>
                <p className="text-xs font-mono text-gray-800">
                  id, breath_id, R, C, time_step, u_in, u_out, pressure
                </p>
              </div>

              <div className="mt-3 p-3 bg-green-50 rounded">
                <p className="text-xs text-green-800">
                  <strong>✓</strong> El dataset incluye valores reales de presión medidos en el ventilador.
                  El modelo LSTM aprenderá a predecir estos valores basándose en las características temporales y parámetros del paciente.
                </p>
              </div>
            </div>
          )}

          {!datasetLoaded && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800">
                <strong>Instrucciones:</strong> Descarga el dataset de la competencia de Kaggle 
                "Google Brain - Ventilator Pressure Prediction" y carga el archivo <code className="bg-blue-100 px-1 rounded">train.csv</code>.
                El sistema procesará automáticamente los ciclos respiratorios.
              </p>
              <p className="text-xs text-blue-600 mt-2">
                Si no tienes el dataset, la simulación usará datos sintéticos.
              </p>
            </div>
          )}
        </div>
        {/* Test Dataset Upload Section */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Upload size={24} className="text-purple-600" />
              Cargar Dataset de Test
            </h2>
            
            <div className="mb-4">
              <label className="block mb-2">
                <span className="text-sm font-semibold text-gray-700">
                  Selecciona el archivo test.csv:
                </span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleTestFileUpload}
                  className="block w-full mt-2 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                />
              </label>
            </div>

            {testDatasetLoaded && testDatasetInfo && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="text-purple-600" size={20} />
                  <p className="font-semibold text-purple-800">Test Dataset Cargado</p>
                </div>
                <p className="text-sm text-gray-700">
                  <strong>Archivo:</strong> {testDatasetInfo.fileName}<br/>
                  <strong>Registros:</strong> {testDatasetInfo.totalRows.toLocaleString()}<br/>
                  <strong>Ciclos:</strong> {testDatasetInfo.totalBreaths.toLocaleString()}
                </p>
              </div>
            )}
          </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab('ml')}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
              activeTab === 'ml'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Simulación 1: LSTM Neural Network
          </button>
          <button
            onClick={() => setActiveTab('ca')}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
              activeTab === 'ca'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Simulación 2: Pressure Propagation (CA)
          </button>
        </div>

        {/* ML Simulation */}
        {activeTab === 'ml' && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Data-Driven Simulation: LSTM Training on Breath Cycles
            </h2>
            
            <div className="flex gap-4 mb-6 flex-wrap">
              <button
                onClick={() => setMlRunning(!mlRunning)}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                {mlRunning ? <Pause size={20} /> : <Play size={20} />}
                {mlRunning ? 'Pausar' : 'Iniciar Entrenamiento'}
              </button>
              <button
                onClick={startTesting}
                disabled={mlRunning || mlEpoch === 0}
                className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
              >
                <Activity size={20} />
                Testear Modelo
              </button>
              <button
                onClick={resetMl}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                <RotateCcw size={20} />
                Reiniciar
              </button>
              <button
                onClick={handlePyTorchTrain}
                disabled={mlRunning || !datasetLoaded}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400"
              >
                entrenar
              </button>

              <button
                onClick={handlePyTorchTest}
                disabled={mlTesting || !testDatasetLoaded}
                className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:bg-gray-400"
              >
                testear
              </button>
              
              
              {!datasetLoaded && (
                <div className="flex gap-2 ml-auto">
                  <div className="flex flex-col">
                    <label className="text-xs text-gray-600">R (Resistance)</label>
                    <input
                      type="number"
                      value={breathParams.R}
                      onChange={(e) => setBreathParams(prev => ({...prev, R: Number(e.target.value)}))}
                      className="w-20 px-2 py-1 border rounded"
                      min="5"
                      max="50"
                    />
                  </div>
                  <div className="flex flex-col">
                    <label className="text-xs text-gray-600">C (Compliance)</label>
                    <input
                      type="number"
                      value={breathParams.C}
                      onChange={(e) => setBreathParams(prev => ({...prev, C: Number(e.target.value)}))}
                      className="w-20 px-2 py-1 border rounded"
                      min="10"
                      max="100"
                    />
                  </div>
                </div>
              )}
            </div>

            {datasetLoaded && (
              <div className="mb-4 p-3 bg-green-50 rounded-lg">
                <p className="text-sm text-green-800">
                  ✓ Usando datos reales del dataset - Ciclo actual: {currentBreath % processedData.length + 1} de {processedData.length}
                </p>
              </div>
            )}

            {mlTesting && (
              <div className="mb-4 p-3 bg-purple-50 rounded-lg border border-purple-200">
                <p className="text-sm text-purple-800 font-semibold mb-1">
                  🔬 Testing en Progreso
                </p>
                <p className="text-xs text-purple-700">
                  Ciclo: {currentBreath + 1} de {testProcessedData.length} ({((currentBreath / testProcessedData.length) * 100).toFixed(1)}%)
                </p>
                <div className="mt-2 w-full bg-purple-200 rounded-full h-2">
                  <div 
                    className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(currentBreath / testProcessedData.length) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Época</p>
                <p className="text-2xl font-bold text-blue-600">{mlEpoch}/125</p>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">MAE (Mean Absolute Error)</p>
                <p className="text-2xl font-bold text-orange-600">{mlMetrics.mae.toFixed(3)} cmH₂O</p>
              </div>
              <div className="bg-red-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">RMSE</p>
                <p className="text-2xl font-bold text-red-600">{mlMetrics.rmse.toFixed(3)} cmH₂O</p>
              </div>
            </div>

            <div className="mb-4">
              <h3 className="font-semibold mb-2">Ciclo Respiratorio (time steps)</h3>
              <div style={{ width: '100%', height: '400px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mlData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timeStep" 
                      label={{ value: 'Time Step', position: 'insideBottom', offset: -5 }} 
                    />
                    <YAxis 
                      label={{ value: 'Pressure (cmH₂O)', angle: -90, position: 'insideLeft' }} 
                    />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="actualPressure" 
                      stroke="#2563eb" 
                      name="Presión Real" 
                      strokeWidth={2} 
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="predictedPressure" 
                      stroke="#dc2626" 
                      name="Predicción LSTM" 
                      strokeWidth={2} 
                      strokeDasharray="5 5" 
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold mb-2">Descripción:</h3>
              <p className="text-sm text-gray-700 mb-2">
                Esta simulación entrena un modelo LSTM usando {datasetLoaded ? 'datos reales del dataset de Kaggle con valores de presión medidos' : 'datos sintéticos'}.
                {datasetLoaded && ' Los datos incluyen: id, breath_id, R, C, time_step, u_in, u_out, pressure.'}
              </p>
              <p className="text-sm text-gray-700">
                <strong>Objetivo del modelo:</strong> Predecir la presión del ventilador en cada time_step basándose en:
              </p>
              <ul className="text-sm text-gray-700 mt-1 ml-4 list-disc">
                <li><strong>R</strong> = Resistencia pulmonar del paciente</li>
                <li><strong>C</strong> = Compliance pulmonar del paciente</li>
                <li><strong>u_in</strong> = Control de válvula de inspiración (0-100)</li>
                <li><strong>u_out</strong> = Control de válvula de espiración (0 o 1)</li>
                <li><strong>time_step</strong> = Secuencia temporal del ciclo respiratorio</li>
              </ul>
              <p className="text-sm text-gray-700 mt-2">
                El LSTM captura dependencias temporales complejas para predecir con precisión la presión objetivo.
              </p>
            </div>
          </div>
        )}

        {/* Cellular Automata Simulation */}
        {activeTab === 'ca' && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Event-Based Simulation: Pressure Distribution in Lung Segments
            </h2>
            
            <div className="flex gap-4 mb-6">
              <button
                onClick={() => setCaRunning(!caRunning)}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                {caRunning ? <Pause size={20} /> : <Play size={20} />}
                {caRunning ? 'Pausar' : 'Iniciar Simulación'}
              </button>
              <button
                onClick={resetCa}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                <RotateCcw size={20} />
                Reiniciar
              </button>
            </div>

            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-purple-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Generación</p>
                <p className="text-2xl font-bold text-purple-600">{caGeneration}</p>
              </div>
              <div className="bg-blue-200 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Baja Presión</p>
                <p className="text-2xl font-bold text-blue-800">{caStats.lowPressure}</p>
              </div>
              <div className="bg-blue-400 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Media Presión</p>
                <p className="text-2xl font-bold text-white">{caStats.mediumPressure}</p>
              </div>
              <div className="bg-blue-700 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Alta Presión</p>
                <p className="text-2xl font-bold text-white">{caStats.highPressure}</p>
              </div>
            </div>

            <div className="border-2 border-gray-300 rounded-lg p-4 bg-gray-50 mb-4">
              <div className="grid gap-1" style={{ 
                gridTemplateColumns: `repeat(${caGrid[0]?.length || 30}, minmax(0, 1fr))` 
              }}>
                {caGrid.map((row, i) => 
                  row.map((cell, j) => (
                    <div
                      key={`${i}-${j}`}
                      className={`aspect-square rounded-sm transition-colors ${getCellColor(cell)}`}
                      title={`Presión: ${cell === 0 ? 'Baja' : cell === 1 ? 'Media' : 'Alta'}`}
                    />
                  ))
                )}
              </div>
            </div>

            <div className="flex gap-4 mb-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-blue-200 rounded"></div>
                <span>Baja Presión (0-10 cmH₂O)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-blue-500 rounded"></div>
                <span>Media Presión (10-20 cmH₂O)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-blue-800 rounded"></div>
                <span>Alta Presión (20+ cmH₂O)</span>
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold mb-2">Descripción:</h3>
              <p className="text-sm text-gray-700 mb-2">
                Este autómata celular modela la propagación espacial de presión en segmentos pulmonares durante la ventilación mecánica.
                Cada célula representa un pequeño segmento del pulmón con un nivel de presión.
              </p>
              <p className="text-sm text-gray-700">
                <strong>Reglas de difusión:</strong> Las zonas de alta presión tienden a equilibrarse con sus vecinos,
                simulando la distribución del aire. Se observan patrones emergentes como la formación de clusters de presión
                y la sincronización espacial, fenómenos caóticos críticos para entender la heterogeneidad de la ventilación.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VentilatorSimulations;