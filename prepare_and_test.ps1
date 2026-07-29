# -----------------------------------------------------------------
# PART 1: PROJECT SETUP (Run this while Docker is installing)
# -----------------------------------------------------------------
echo "--- Step 1: Cleaning up and structuring your project... ---"

# Create the required 'models' and 'templates' directories if they don't exist
if (-not (Test-Path -Path "models" -PathType Container)) {
    mkdir models
    echo "Created 'models' directory."
}
if (-not (Test-Path -Path "templates" -PathType Container)) {
    mkdir templates
    echo "Created 'templates' directory."
}

# Move all model files into the 'models' directory
echo "Moving model files..."
Move-Item -Path "*.h5" -Destination "models/" -ErrorAction SilentlyContinue
Move-Item -Path "*.joblib" -Destination "models/" -ErrorAction SilentlyContinue

# Move the HTML file (assuming it's named index.html) into the 'templates' directory
echo "Moving HTML template..."
Move-Item -Path "index.html" -Destination "templates/" -ErrorAction SilentlyContinue

# Delete the problematic .gcloudignore file to ensure all files are uploaded for cloud builds
echo "Removing .gcloudignore file..."
if (Test-Path -Path ".gcloudignore") {
    Remove-Item .gcloudignore
    echo "Successfully removed .gcloudignore."
}

echo "✅ Project setup is complete. Your folder is now correctly structured."
echo ""
echo "-----------------------------------------------------------------"
echo "PART 2: LOCAL DOCKER TEST (Run this after Docker is running)"
echo "-----------------------------------------------------------------"
echo "Once Docker Desktop is installed and running, use the following commands:"
echo ""
echo "Build the container image with:"
echo "docker build -t neurovocal-final ."
echo ""
echo "Then, run the container with:"
echo "docker run --rm -p 8080:8080 -e PORT=8080 neurovocal-final"
echo ""
```# -----------------------------------------------------------------
# PART 1: PROJECT SETUP (Run this while Docker is installing)
# -----------------------------------------------------------------
echo "--- Step 1: Cleaning up and structuring your project... ---"

# Create the required 'models' and 'templates' directories if they don't exist
if (-not (Test-Path -Path "models" -PathType Container)) {
    mkdir models
    echo "Created 'models' directory."
}
if (-not (Test-Path -Path "templates" -PathType Container)) {
    mkdir templates
    echo "Created 'templates' directory."
}

# Move all model files into the 'models' directory
echo "Moving model files..."
Move-Item -Path "*.h5" -Destination "models/" -ErrorAction SilentlyContinue
Move-Item -Path "*.joblib" -Destination "models/" -ErrorAction SilentlyContinue

# Move the HTML file (assuming it's named index.html) into the 'templates' directory
echo "Moving HTML template..."
Move-Item -Path "index.html" -Destination "templates/" -ErrorAction SilentlyContinue

# Delete the problematic .gcloudignore file to ensure all files are uploaded for cloud builds
echo "Removing .gcloudignore file..."
if (Test-Path -Path ".gcloudignore") {
    Remove-Item .gcloudignore
    echo "Successfully removed .gcloudignore."
}

echo "✅ Project setup is complete. Your folder is now correctly structured."
echo ""
echo "-----------------------------------------------------------------"
echo "PART 2: LOCAL DOCKER TEST (Run this after Docker is running)"
echo "-----------------------------------------------------------------"
echo "Once Docker Desktop is installed and running, use the following commands:"
echo ""
echo "Build the container image with:"
echo "docker build -t neurovocal-final ."
echo ""
echo "Then, run the container with:"
echo "docker run --rm -p 8080:8080 -e PORT=8080 neurovocal-final"
echo ""
