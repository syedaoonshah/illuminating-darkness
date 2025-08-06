/**
 * NIQE Calculator Library
 * Natural Image Quality Evaluator - No-Reference Image Quality Assessment
 * 
 * Based on the UTLIVE NIQE algorithm
 * Reference: "Making a "Completely Blind" Image Quality Analyzer" by Mittal et al.
 */

class NIQECalculator {
    constructor() {
        console.log('🎯 NIQE Calculator initialized');
        
        // Pre-trained model parameters (these would typically be loaded from a trained model)
        this.modelParams = null;
        this.isModelLoaded = false;
        
        // Initialize with default parameters
        this.initializeDefaultModel();
    }

    /**
     * Initialize with default pre-trained model parameters
     * In a real implementation, these would be loaded from a .mat file or trained data
     */
    initializeDefaultModel() {
        // These are simplified default parameters
        // Real NIQE uses pre-trained model from natural images
        this.modelParams = {
            // Mean and covariance of natural scene statistics features
            mu: new Array(36).fill(0).map(() => Math.random() * 0.1),
            cov: this.generateDefaultCovariance(36),
            // Patch size and overlap
            patchSize: 96,
            overlap: 0,
            // Filter parameters
            gamma: 0.2,
            sigma: 7/6
        };
        
        this.isModelLoaded = true;
        console.log('📊 Default NIQE model parameters initialized');
    }

    /**
     * Generate a default covariance matrix
     * @param {number} size - Matrix size
     * @returns {Array<Array<number>>} Covariance matrix
     */
    generateDefaultCovariance(size) {
        const cov = [];
        for (let i = 0; i < size; i++) {
            cov[i] = [];
            for (let j = 0; j < size; j++) {
                if (i === j) {
                    cov[i][j] = 1.0; // Diagonal elements
                } else {
                    cov[i][j] = Math.random() * 0.1; // Small correlation
                }
            }
        }
        return cov;
    }

    /**
     * Main NIQE calculation function
     * @param {ImageData} imageData - Canvas ImageData of the image
     * @returns {Promise<number>} NIQE score (lower is better)
     */
    async computeNIQE(imageData) {
        if (!this.isModelLoaded) {
            throw new Error('NIQE model not loaded');
        }

        try {
            // Convert to grayscale
            const grayImage = this.rgbToGray(imageData);
            
            // Extract patches
            const patches = this.extractPatches(grayImage, this.modelParams.patchSize);
            
            // Compute features for each patch
            const features = [];
            for (const patch of patches) {
                const patchFeatures = this.computePatchFeatures(patch);
                if (patchFeatures) {
                    features.push(patchFeatures);
                }
            }
            
            if (features.length === 0) {
                throw new Error('No valid patches found for NIQE calculation');
            }
            
            // Compute mean feature vector
            const meanFeatures = this.computeMeanFeatures(features);
            
            // Calculate NIQE score using Mahalanobis distance
            const niqeScore = this.computeMahalanobisDistance(meanFeatures, this.modelParams.mu, this.modelParams.cov);
            
            return niqeScore;
            
        } catch (error) {
            console.error('Error in NIQE computation:', error);
            throw error;
        }
    }

    /**
     * Convert RGB image data to grayscale
     * @param {ImageData} imageData - Canvas ImageData
     * @returns {Array<Array<number>>} 2D grayscale array
     */
    rgbToGray(imageData) {
        const { data, width, height } = imageData;
        const gray = [];
        
        for (let i = 0; i < height; i++) {
            gray[i] = [];
            for (let j = 0; j < width; j++) {
                const idx = (i * width + j) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                // Use standard luminance formula
                gray[i][j] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        }
        return gray;
    }

    /**
     * Extract non-overlapping patches from image
     * @param {Array<Array<number>>} image - 2D grayscale image
     * @param {number} patchSize - Size of patches to extract
     * @returns {Array<Array<Array<number>>>} Array of patches
     */
    extractPatches(image, patchSize) {
        const patches = [];
        const height = image.length;
        const width = image[0].length;
        
        for (let i = 0; i <= height - patchSize; i += patchSize) {
            for (let j = 0; j <= width - patchSize; j += patchSize) {
                const patch = [];
                for (let pi = 0; pi < patchSize; pi++) {
                    patch[pi] = [];
                    for (let pj = 0; pj < patchSize; pj++) {
                        patch[pi][pj] = image[i + pi][j + pj];
                    }
                }
                patches.push(patch);
            }
        }
        
        console.log(`📐 Extracted ${patches.length} patches of size ${patchSize}×${patchSize}`);
        return patches;
    }

    /**
     * Compute NIQE features for a single patch
     * @param {Array<Array<number>>} patch - 2D patch array
     * @returns {Array<number>|null} Feature vector or null if patch is invalid
     */
    computePatchFeatures(patch) {
        try {
            const features = [];
            
            // Compute local mean
            const localMean = this.computeMean(patch);
            
            // Compute local variance
            const localVar = this.computeVariance(patch, localMean);
            
            // Skip patches with very low variance (flat regions)
            if (localVar < 1e-6) {
                return null;
            }
            
            // Normalize patch
            const normalizedPatch = this.normalizePatch(patch, localMean, Math.sqrt(localVar));
            
            // Compute Generalized Gaussian Distribution (GGD) parameters
            const ggdParams = this.fitGGD(this.flatten2D(normalizedPatch));
            features.push(ggdParams.alpha);
            features.push(ggdParams.sigma);
            
            // Compute features from horizontal, vertical, and diagonal differences
            const hDiff = this.computeHorizontalDifferences(normalizedPatch);
            const vDiff = this.computeVerticalDifferences(normalizedPatch);
            const d1Diff = this.computeDiagonal1Differences(normalizedPatch);
            const d2Diff = this.computeDiagonal2Differences(normalizedPatch);
            
            // Fit GGD to difference images
            const hGGD = this.fitGGD(this.flatten2D(hDiff));
            const vGGD = this.fitGGD(this.flatten2D(vDiff));
            const d1GGD = this.fitGGD(this.flatten2D(d1Diff));
            const d2GGD = this.fitGGD(this.flatten2D(d2Diff));
            
            features.push(hGGD.alpha, hGGD.sigma);
            features.push(vGGD.alpha, vGGD.sigma);
            features.push(d1GGD.alpha, d1GGD.sigma);
            features.push(d2GGD.sigma, d2GGD.sigma);
            
            // Compute asymmetric generalized Gaussian distribution (AGGD) parameters
            const hAGGD = this.fitAGGD(this.flatten2D(hDiff));
            const vAGGD = this.fitAGGD(this.flatten2D(vDiff));
            const d1AGGD = this.fitAGGD(this.flatten2D(d1Diff));
            const d2AGGD = this.fitAGGD(this.flatten2D(d2Diff));
            
            features.push(hAGGD.eta, hAGGD.nu);
            features.push(vAGGD.eta, vAGGD.nu);
            features.push(d1AGGD.eta, d1AGGD.nu);
            features.push(d2AGGD.eta, d2AGGD.nu);
            
            // Pad or truncate to expected feature length
            while (features.length < 36) {
                features.push(0);
            }
            
            return features.slice(0, 36);
            
        } catch (error) {
            console.warn('Error computing patch features:', error);
            return null;
        }
    }

    /**
     * Compute mean of 2D array
     * @param {Array<Array<number>>} arr - 2D array
     * @returns {number} Mean value
     */
    computeMean(arr) {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < arr.length; i++) {
            for (let j = 0; j < arr[i].length; j++) {
                sum += arr[i][j];
                count++;
            }
        }
        return sum / count;
    }

    /**
     * Compute variance of 2D array
     * @param {Array<Array<number>>} arr - 2D array
     * @param {number} mean - Pre-computed mean
     * @returns {number} Variance
     */
    computeVariance(arr, mean) {
        let sumSquares = 0;
        let count = 0;
        for (let i = 0; i < arr.length; i++) {
            for (let j = 0; j < arr[i].length; j++) {
                const diff = arr[i][j] - mean;
                sumSquares += diff * diff;
                count++;
            }
        }
        return sumSquares / (count - 1);
    }

    /**
     * Normalize patch by subtracting mean and dividing by standard deviation
     * @param {Array<Array<number>>} patch - Input patch
     * @param {number} mean - Mean to subtract
     * @param {number} std - Standard deviation to divide by
     * @returns {Array<Array<number>>} Normalized patch
     */
    normalizePatch(patch, mean, std) {
        const normalized = [];
        for (let i = 0; i < patch.length; i++) {
            normalized[i] = [];
            for (let j = 0; j < patch[i].length; j++) {
                normalized[i][j] = (patch[i][j] - mean) / std;
            }
        }
        return normalized;
    }

    /**
     * Flatten 2D array to 1D
     * @param {Array<Array<number>>} arr2d - 2D array
     * @returns {Array<number>} 1D array
     */
    flatten2D(arr2d) {
        const flat = [];
        for (let i = 0; i < arr2d.length; i++) {
            for (let j = 0; j < arr2d[i].length; j++) {
                flat.push(arr2d[i][j]);
            }
        }
        return flat;
    }

    /**
     * Compute horizontal differences
     * @param {Array<Array<number>>} patch - Input patch
     * @returns {Array<Array<number>>} Horizontal differences
     */
    computeHorizontalDifferences(patch) {
        const diff = [];
        for (let i = 0; i < patch.length; i++) {
            diff[i] = [];
            for (let j = 0; j < patch[i].length - 1; j++) {
                diff[i][j] = patch[i][j + 1] - patch[i][j];
            }
        }
        return diff;
    }

    /**
     * Compute vertical differences
     * @param {Array<Array<number>>} patch - Input patch
     * @returns {Array<Array<number>>} Vertical differences
     */
    computeVerticalDifferences(patch) {
        const diff = [];
        for (let i = 0; i < patch.length - 1; i++) {
            diff[i] = [];
            for (let j = 0; j < patch[i].length; j++) {
                diff[i][j] = patch[i + 1][j] - patch[i][j];
            }
        }
        return diff;
    }

    /**
     * Compute diagonal differences (top-left to bottom-right)
     * @param {Array<Array<number>>} patch - Input patch
     * @returns {Array<Array<number>>} Diagonal differences
     */
    computeDiagonal1Differences(patch) {
        const diff = [];
        for (let i = 0; i < patch.length - 1; i++) {
            diff[i] = [];
            for (let j = 0; j < patch[i].length - 1; j++) {
                diff[i][j] = patch[i + 1][j + 1] - patch[i][j];
            }
        }
        return diff;
    }

    /**
     * Compute diagonal differences (top-right to bottom-left)
     * @param {Array<Array<number>>} patch - Input patch
     * @returns {Array<Array<number>>} Diagonal differences
     */
    computeDiagonal2Differences(patch) {
        const diff = [];
        for (let i = 0; i < patch.length - 1; i++) {
            diff[i] = [];
            for (let j = 1; j < patch[i].length; j++) {
                diff[i][j - 1] = patch[i + 1][j - 1] - patch[i][j];
            }
        }
        return diff;
    }

    /**
     * Fit Generalized Gaussian Distribution (GGD) parameters
     * @param {Array<number>} data - 1D data array
     * @returns {Object} GGD parameters {alpha, sigma}
     */
    fitGGD(data) {
        if (data.length === 0) return { alpha: 1, sigma: 1 };
        
        // Simple moment-based estimation (simplified version)
        const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
        const variance = data.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) / data.length;
        
        // Estimate alpha and sigma using method of moments (simplified)
        const alpha = Math.max(0.1, Math.min(2.0, Math.sqrt(variance) / 2));
        const sigma = Math.max(0.01, Math.sqrt(variance));
        
        return { alpha, sigma };
    }

    /**
     * Fit Asymmetric Generalized Gaussian Distribution (AGGD) parameters
     * @param {Array<number>} data - 1D data array
     * @returns {Object} AGGD parameters {eta, nu}
     */
    fitAGGD(data) {
        if (data.length === 0) return { eta: 1, nu: 1 };
        
        // Separate positive and negative values
        const positive = data.filter(x => x > 0);
        const negative = data.filter(x => x < 0).map(x => -x);
        
        // Simplified parameter estimation
        const posVar = positive.length > 0 ? 
            positive.reduce((sum, val) => sum + val * val, 0) / positive.length : 1;
        const negVar = negative.length > 0 ? 
            negative.reduce((sum, val) => sum + val * val, 0) / negative.length : 1;
        
        const eta = Math.max(0.01, Math.sqrt(posVar));
        const nu = Math.max(0.01, Math.sqrt(negVar));
        
        return { eta, nu };
    }

    /**
     * Compute mean feature vector from all patches
     * @param {Array<Array<number>>} features - Array of feature vectors
     * @returns {Array<number>} Mean feature vector
     */
    computeMeanFeatures(features) {
        const numFeatures = features[0].length;
        const meanFeatures = new Array(numFeatures).fill(0);
        
        for (let i = 0; i < features.length; i++) {
            for (let j = 0; j < numFeatures; j++) {
                meanFeatures[j] += features[i][j];
            }
        }
        
        for (let j = 0; j < numFeatures; j++) {
            meanFeatures[j] /= features.length;
        }
        
        return meanFeatures;
    }

    /**
     * Compute Mahalanobis distance between feature vectors
     * @param {Array<number>} x - Feature vector
     * @param {Array<number>} mu - Mean vector
     * @param {Array<Array<number>>} cov - Covariance matrix
     * @returns {number} Mahalanobis distance (NIQE score)
     */
    computeMahalanobisDistance(x, mu, cov) {
        try {
            // Compute difference vector
            const diff = x.map((val, i) => val - mu[i]);
            
            // Compute inverse covariance (simplified - use diagonal approximation)
            const invCov = this.computeDiagonalInverse(cov);
            
            // Compute Mahalanobis distance: sqrt((x-mu)' * inv(cov) * (x-mu))
            let distance = 0;
            for (let i = 0; i < diff.length; i++) {
                distance += diff[i] * invCov[i] * diff[i];
            }
            
            return Math.sqrt(Math.abs(distance));
            
        } catch (error) {
            console.error('Error computing Mahalanobis distance:', error);
            // Fallback to Euclidean distance
            const diff = x.map((val, i) => val - mu[i]);
            return Math.sqrt(diff.reduce((sum, val) => sum + val * val, 0));
        }
    }

    /**
     * Compute diagonal inverse of covariance matrix (simplified approach)
     * @param {Array<Array<number>>} cov - Covariance matrix
     * @returns {Array<number>} Diagonal of inverse covariance
     */
    computeDiagonalInverse(cov) {
        const diag = [];
        for (let i = 0; i < cov.length; i++) {
            const val = cov[i][i];
            diag[i] = val > 1e-10 ? 1.0 / val : 1.0;
        }
        return diag;
    }

    /**
     * Calculate NIQE for an Image object
     * @param {Image} image - HTML Image object
     * @returns {Promise<number>} NIQE score
     */
    async calculateImageNIQE(image) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, image.width, image.height);
        return await this.computeNIQE(imageData);
    }

    /**
     * Get performance interpretation for NIQE score
     * @param {number} niqeScore - NIQE score
     * @returns {string} Performance interpretation
     */
    getInterpretation(niqeScore) {
        if (niqeScore < 3) {
            return 'Excellent image quality - very natural appearance';
        } else if (niqeScore < 4) {
            return 'Good image quality - natural appearance with minor artifacts';
        } else if (niqeScore < 6) {
            return 'Fair image quality - noticeable but acceptable artifacts';
        } else {
            return 'Poor image quality - significant artifacts affecting naturalness';
        }
    }

    /**
     * Get performance rating for NIQE score
     * @param {number} niqeScore - NIQE score
     * @returns {string} Performance rating
     */
    getPerformanceRating(niqeScore) {
        if (niqeScore < 3) return 'Excellent';
        if (niqeScore < 4) return 'Good';
        if (niqeScore < 6) return 'Fair';
        return 'Poor';
    }

    /**
     * Get color for performance rating
     * @param {string} rating - Performance rating
     * @returns {string} CSS color
     */
    getRatingColor(rating) {
        const colors = {
            'Excellent': '#4caf50',
            'Good': '#8bc34a',
            'Fair': '#ff9800',
            'Poor': '#f44336'
        };
        return colors[rating] || '#666';
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NIQECalculator;
} else if (typeof window !== 'undefined') {
    window.NIQECalculator = NIQECalculator;
}
