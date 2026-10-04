import { useRef, useState } from "react";
import axios from "axios";

import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

  const defectNames = {
    Cr: "Crazing",
    In: "Inclusion",
    Pa: "Patches",
    PS: "Pitted Surface",
    RS: "Rolled-in Scale",
    Sc: "Scratches",
  };

  const formatDefectName = (defect) => {
    if (!defect) {
      return "Unknown";
    }

    return defectNames[defect] || defect;
  };

  const getConfidenceClass = (confidence) => {
    const value = Number(confidence);

    if (value >= 80) {
      return "confidence-high";
    }

    if (value >= 60) {
      return "confidence-medium";
    }

    return "confidence-low";
  };

  const processFile = (file) => {
    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      setSelectedFile(null);
      setPreview(null);
      setResult(null);
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setError(null);

    const imageUrl = URL.createObjectURL(file);
    setPreview(imageUrl);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    processFile(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files[0];
    processFile(file);
  };

  const openFileSelector = () => {
    fileInputRef.current?.click();
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select a steel image first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      if (!response.data) {
        throw new Error("Invalid response from prediction server.");
      }

      setResult(response.data);
    } catch (err) {
      console.error("Prediction error:", err);

      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (
        err.message === "Invalid response from prediction server."
      ) {
        setError("The prediction server returned an invalid response.");
      } else {
        setError(
          "Unable to connect to the prediction server. Please make sure the backend is running."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setIsDragging(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getSortedProbabilities = () => {
    if (!result?.probabilities) {
      return [];
    }

    return Object.entries(result.probabilities).sort(
      ([, valueA], [, valueB]) => Number(valueB) - Number(valueA)
    );
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">SV</div>

          <div>
            <div className="brand-name">SteelVision AI</div>
            <div className="brand-subtitle">
              Intelligent Surface Inspection
            </div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          <span>AI System Online</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero-section">
          <div>
            <p className="eyebrow">AI-POWERED QUALITY CONTROL</p>

            <h1>
              Steel Surface
              <span> Defect Inspection</span>
            </h1>

            <p className="hero-description">
              Upload a steel surface image and let the AI model identify
              potential surface defects with confidence-based classification.
            </p>
          </div>
        </section>

        <section className="inspection-grid">
          {/* Upload / Image Panel */}
          <div className="panel upload-panel">
            <div className="panel-header">
              <div>
                <p className="panel-eyebrow">INPUT</p>
                <h2>Steel Image</h2>
              </div>

              {selectedFile && (
                <span className="file-ready">Image Ready</span>
              )}
            </div>

            {!preview ? (
              <div
                className={`drop-zone ${isDragging ? "dragging" : ""}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={openFileSelector}
              >
                <div className="upload-icon">
                  ↑
                </div>

                <h3>Upload Steel Image</h3>

                <p>
                  Drag and drop your image here
                  <br />
                  or click to browse
                </p>

                <span className="supported-formats">
                  BMP • JPG • JPEG • PNG
                </span>

                <input
                  ref={fileInputRef}
                  id="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  hidden
                />

                <button
                  type="button"
                  className="browse-button"
                  onClick={(event) => {
                    event.stopPropagation();
                    openFileSelector();
                  }}
                >
                  Choose Image
                </button>
              </div>
            ) : (
              <div className="image-preview-wrapper">
                <div className="image-preview">
                  <img
                    src={preview}
                    alt="Selected steel surface"
                  />

                  <div className="image-overlay">
                    <span>Selected Image</span>
                  </div>
                </div>

                <div className="file-information">
                  <div className="file-icon">IMG</div>

                  <div className="file-details">
                    <strong>{selectedFile.name}</strong>

                    <span>
                      {(selectedFile.size / 1024).toFixed(1)} KB
                    </span>
                  </div>

                  <button
                    type="button"
                    className="change-button"
                    onClick={openFileSelector}
                  >
                    Change
                  </button>
                </div>

                <input
                  ref={fileInputRef}
                  id="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  hidden
                />
              </div>
            )}

            <div className="action-buttons">
              <button
                className="predict-button"
                onClick={handlePredict}
                disabled={loading || !selectedFile}
              >
                {loading ? (
                  <>
                    <span className="button-spinner"></span>
                    Analyzing Surface...
                  </>
                ) : (
                  <>
                    <span>✦</span>
                    Analyze Surface
                  </>
                )}
              </button>

              <button
                className="reset-button"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            </div>
          </div>

          {/* Result Panel */}
          <div className="panel result-panel">
            <div className="panel-header">
              <div>
                <p className="panel-eyebrow">AI ANALYSIS</p>
                <h2>Inspection Result</h2>
              </div>

              {result && (
                <span className="completed-badge">
                  ✓ Complete
                </span>
              )}
            </div>

            {!result && !loading && !error && (
              <div className="empty-result">
                <div className="result-placeholder-icon">◉</div>

                <h3>Awaiting Inspection</h3>

                <p>
                  Upload a steel image and click{" "}
                  <strong>Analyze Surface</strong> to see the AI prediction.
                </p>
              </div>
            )}

            {loading && (
              <div className="analysis-loading">
                <div className="large-spinner"></div>

                <h3>Analyzing Surface</h3>

                <p>
                  The ResNet18 model is processing the uploaded image.
                </p>

                <div className="loading-line">
                  <span></span>
                </div>
              </div>
            )}

            {error && !loading && (
              <div className="error-box">
                <div className="error-icon">!</div>

                <div>
                  <strong>Prediction Error</strong>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {result && !loading && (
              <div className="result-content">
                <div className="prediction-label">
                  DETECTED DEFECT
                </div>

                <div className="prediction-name">
                  {formatDefectName(result.predicted_defect)}
                </div>

                <div className="prediction-code">
                  Class Code: {result.predicted_defect}
                </div>

                <div className="confidence-card">
                  <div className="confidence-header">
                    <span>Model Confidence</span>

                    <strong>
                      {Number(result.confidence).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="confidence-bar">
                    <div
                      className={`confidence-fill ${getConfidenceClass(
                        result.confidence
                      )}`}
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(0, Number(result.confidence))
                        )}%`,
                      }}
                    ></div>
                  </div>

                  <div className="confidence-scale">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div className="result-summary">
                  <div>
                    <span>Classification</span>
                    <strong>Surface Defect</strong>
                  </div>

                  <div>
                    <span>Model</span>
                    <strong>ResNet18</strong>
                  </div>

                  <div>
                    <span>Classes</span>
                    <strong>6</strong>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Probability Section */}
        {result?.probabilities && (
          <section className="panel probability-panel">
            <div className="section-heading">
              <div>
                <p className="panel-eyebrow">MODEL OUTPUT</p>
                <h2>Class Probabilities</h2>
              </div>

              <span className="probability-count">
                6 defect classes
              </span>
            </div>

            <div className="probability-grid">
              {getSortedProbabilities().map(
                ([className, probability], index) => {
                  const percentage = Number(probability);

                  return (
                    <div
                      className={`probability-item ${
                        index === 0 ? "top-prediction" : ""
                      }`}
                      key={className}
                    >
                      <div className="probability-top">
                        <div className="probability-name">
                          <span className="rank-number">
                            {index + 1}
                          </span>

                          <span>
                            {formatDefectName(className)}
                          </span>
                        </div>

                        <strong>
                          {percentage.toFixed(2)}%
                        </strong>
                      </div>

                      <div className="probability-bar">
                        <div
                          className="probability-fill"
                          style={{
                            width: `${Math.min(
                              100,
                              Math.max(0, percentage)
                            )}%`,
                          }}
                        ></div>
                      </div>
                    </div>
                  );
                }
              )}
            </div>
          </section>
        )}

        {/* Model Information */}
        <section className="model-info">
          <div className="info-card">
            <span className="info-icon">AI</span>

            <div>
              <span>AI MODEL</span>
              <strong>ResNet18</strong>
            </div>
          </div>

          <div className="info-card">
            <span className="info-icon">6</span>

            <div>
              <span>DEFECT CLASSES</span>
              <strong>NEU-CLS</strong>
            </div>
          </div>

          <div className="info-card">
            <span className="info-icon">87%</span>

            <div>
              <span>MODEL ACCURACY</span>
              <strong>87.41%</strong>
            </div>
          </div>

          <div className="info-card">
            <span className="info-icon">API</span>

            <div>
              <span>PROCESSING</span>
              <strong>FastAPI</strong>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span>SteelVision AI</span>
        <span>AI-Based Steel Surface Quality Control System</span>
      </footer>
    </div>
  );
}

export default App;