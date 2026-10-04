import { useState } from "react";
import axios from "axios";

import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    // Validate image
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

    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
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
      } else if (err.message === "Invalid response from prediction server.") {
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

    // Reset file input
    const fileInput = document.getElementById("file-input");

    if (fileInput) {
      fileInput.value = "";
    }
  };

  const formatDefectName = (defect) => {
    if (!defect) {
      return "Unknown";
    }

    const defectNames = {
      Cr: "Crazing",
      In: "Inclusion",
      Pa: "Patches",
      PS: "Pitted Surface",
      RS: "Rolled-in Scale",
      Sc: "Scratches",
    };

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

  return (
    <div className="app">
      <div className="container">

        <h1>Steel Quality Control</h1>

        <p className="subtitle">
          AI-powered steel surface defect detection
        </p>

        <div className="upload-section">

          {/* File Selection */}
          <label className="file-label">
            Select Steel Image

            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
          </label>

          {/* Selected Filename */}
          {selectedFile && (
            <p className="filename">
              Selected: {selectedFile.name}
            </p>
          )}

          {/* Image Preview */}
          {preview && (
            <div className="preview-container">

              <h3>Image Preview</h3>

              <img
                src={preview}
                alt="Steel defect preview"
              />

            </div>
          )}

          {/* Buttons */}
          <div className="button-group">

            <button
              onClick={handlePredict}
              disabled={loading || !selectedFile}
            >
              {loading ? "Analyzing..." : "Predict Defect"}
            </button>

            <button
              className="reset-button"
              onClick={handleReset}
              disabled={loading && !selectedFile}
            >
              Clear
            </button>

          </div>

          {/* Loading */}
          {loading && (
            <div className="loading">
              <div className="spinner"></div>

              <p>
                Analyzing steel surface...
              </p>

              <span>
                Please wait while the AI model processes the image.
              </span>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="error">
              <strong>Prediction Error</strong>
              <p>{error}</p>
            </div>
          )}

          {/* Result */}
          {result && !loading && (
            <div className="result">

              <h2>Prediction Result</h2>

              <div className="result-card">

                <div className="defect-name">
                  {formatDefectName(result.predicted_defect)}
                </div>

                <p className="defect-code">
                  Class: {result.predicted_defect}
                </p>

                <div className="confidence-section">

                  <div className="confidence-header">
                    <span>Confidence</span>

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

                </div>

              </div>

              {/* Probability Visualization */}
              {result.probabilities && (
                <div className="probabilities">

                  <h3>Class Probabilities</h3>

                  {Object.entries(result.probabilities).map(
                    ([className, probability]) => {

                      const percentage = Number(probability);

                      return (
                        <div
                          className="probability-row"
                          key={className}
                        >

                          <div className="probability-label">
                            <span>
                              {formatDefectName(className)}
                            </span>

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
              )}

            </div>
          )}

        </div>

      </div>
    </div>
  );
}

export default App;
