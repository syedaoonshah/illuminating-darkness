/**
 * LOE (Lightness Order Error) Algorithm Implementation
 * 
 * This module implements the exact MATLAB LOE algorithm for image enhancement quality assessment.
 * Based on the original MATLAB code that measures how well enhanced images preserve 
 * the lightness order relationships of the original images.
 * 
 * @author Converted from MATLAB implementation
 * @version 1.0
 */

class LOEAlgorithm {
    /**
     * Main LOE calculation function
     * @param {ImageData} inputData - Canvas ImageData of the input image
     * @param {ImageData} enhancedData - Canvas ImageData of the enhanced image
     * @returns {number} LOE score (lower is better)
     */
    static calculateLOE(inputData, enhancedData) {
        const { width, height } = inputData;
        const m = height; // rows
        const n = width;  // columns
        
        console.log(`🧮 LOE Algorithm: Processing ${n}x${m} images`);
        
        // Convert to grayscale and get local maxima
        const win = 7;
        console.log('   📊 Converting to grayscale...');
        const inputGray = LOEAlgorithm.rgbToGray(inputData);
        const enhancedGray = LOEAlgorithm.rgbToGray(enhancedData);
        
        console.log('   🔍 Computing local maxima...');
        const imax = LOEAlgorithm.getLocalMax(inputGray, n, m, win);
        const emax = LOEAlgorithm.getLocalMax(enhancedGray, n, m, win);
        
        // Downsample
        const blkwin = 50;
        const mind = Math.min(m, n);
        const step = Math.max(1, Math.floor(mind / blkwin));
        const blkm = Math.floor(m / step);
        const blkn = Math.floor(n / step);
        
        console.log(`   ⬇️ Downsampling: step=${step}, blocks=${blkn}x${blkm}`);
        
        if (blkm <= 0 || blkn <= 0) {
            throw new Error(`Invalid block dimensions: ${blkn}x${blkm}`);
        }
        
        const ipic_ds = [];
        const epic_ds = [];
        
        for (let i = 0; i < blkm; i++) {
            ipic_ds[i] = [];
            epic_ds[i] = [];
            for (let j = 0; j < blkn; j++) {
                const row = Math.min(i * step, m - 1);
                const col = Math.min(j * step, n - 1);
                ipic_ds[i][j] = imax[row][col];
                epic_ds[i][j] = emax[row][col];
            }
        }
        
        console.log('   🔢 Computing LOE values...');
        // Calculate LOE
        let totalLOE = 0;
        for (let i = 0; i < blkm; i++) {
            for (let j = 0; j < blkn; j++) {
                let loe = 0;
                for (let p = 0; p < blkm; p++) {
                    for (let q = 0; q < blkn; q++) {
                        const flag1 = ipic_ds[p][q] >= ipic_ds[i][j];
                        const flag2 = epic_ds[p][q] >= epic_ds[i][j];
                        if (flag1 !== flag2) {
                            loe++;
                        }
                    }
                }
                totalLOE += loe;
            }
        }
        
        const finalLOE = totalLOE / (blkm * blkn);
        console.log(`   ✅ Final LOE: ${finalLOE.toFixed(2)}`);
        return finalLOE;
    }

    /**
     * Convert RGB ImageData to grayscale using max(R,G,B) method
     * @param {ImageData} imageData - Canvas ImageData object
     * @returns {number[][]} 2D array of grayscale values
     */
    static rgbToGray(imageData) {
        const { data, width, height } = imageData;
        const gray = [];
        
        for (let i = 0; i < height; i++) {
            gray[i] = [];
            for (let j = 0; j < width; j++) {
                const idx = (i * width + j) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                // Use max(R,G,B) as in the original MATLAB code
                gray[i][j] = Math.max(r, g, b);
            }
        }
        return gray;
    }

    /**
     * Get local maximum for each pixel using a sliding window
     * @param {number[][]} image - 2D grayscale image array
     * @param {number} width - Image width
     * @param {number} height - Image height  
     * @param {number} win - Window size (half-width)
     * @returns {number[][]} 2D array of local maximum values
     */
    static getLocalMax(image, width, height, win) {
        const extImage = LOEAlgorithm.extendImage(image, width, height, win);
        const output = [];
        
        for (let i = 0; i < height; i++) {
            output[i] = [];
            for (let j = 0; j < width; j++) {
                let max = 0;
                for (let di = -win; di <= win; di++) {
                    for (let dj = -win; dj <= win; dj++) {
                        const extRow = i + win + di;
                        const extCol = j + win + dj;
                        if (extImage[extRow] && extImage[extRow][extCol] !== undefined) {
                            const val = extImage[extRow][extCol];
                            if (val > max) max = val;
                        }
                    }
                }
                output[i][j] = max;
            }
        }
        return output;
    }

    /**
     * Extend image boundaries by reflection (mirror padding)
     * @param {number[][]} image - 2D image array
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @param {number} winSize - Extension size
     * @returns {number[][]} Extended image array
     */
    static extendImage(image, width, height, winSize) {
        const extended = [];
        const newHeight = height + 2 * winSize;
        const newWidth = width + 2 * winSize;
        
        // Initialize extended image
        for (let i = 0; i < newHeight; i++) {
            extended[i] = new Array(newWidth).fill(0);
        }
        
        // Copy original image to center
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                extended[i + winSize][j + winSize] = image[i][j];
            }
        }
        
        // Extend edges by reflection (mirror padding)
        // Extend top and bottom edges
        for (let i = 0; i < winSize; i++) {
            for (let j = winSize; j < winSize + width; j++) {
                extended[winSize - 1 - i][j] = extended[winSize + 1 + i][j]; // top
                extended[height + winSize + i][j] = extended[height + winSize - 2 - i][j]; // bottom
            }
        }
        
        // Extend left and right edges
        for (let i = 0; i < newHeight; i++) {
            for (let j = 0; j < winSize; j++) {
                extended[i][winSize - 1 - j] = extended[i][winSize + 1 + j]; // left
                extended[i][width + winSize + j] = extended[i][width + winSize - 2 - j]; // right
            }
        }
        
        return extended;
    }

    /**
     * Async wrapper for LOE calculation to prevent UI blocking
     * @param {ImageData} inputData - Input image data
     * @param {ImageData} enhancedData - Enhanced image data
     * @returns {Promise<number>} Promise resolving to LOE score
     */
    static async computeLOEAsync(inputData, enhancedData) {
        return new Promise((resolve, reject) => {
            // Use setTimeout to prevent UI blocking
            setTimeout(() => {
                try {
                    const result = LOEAlgorithm.calculateLOE(inputData, enhancedData);
                    resolve(result);
                } catch (error) {
                    console.error('❌ Error in LOE algorithm:', error);
                    reject(error);
                }
            }, 10); // Small delay to allow UI updates
        });
    }

    /**
     * Utility function to get LOE performance rating
     * @param {number} loeValue - LOE score
     * @returns {string} Performance rating
     */
    static getPerformanceRating(loeValue) {
        if (loeValue < 35) return 'Excellent';
        if (loeValue < 50) return 'Good';
        if (loeValue < 65) return 'Fair';
        return 'Poor';
    }

    /**
     * Utility function to get rating color
     * @param {string} rating - Performance rating
     * @returns {string} CSS color
     */
    static getRatingColor(rating) {
        const colors = {
            'Excellent': '#4caf50',
            'Good': '#8bc34a', 
            'Fair': '#ff9800',
            'Poor': '#f44336'
        };
        return colors[rating] || '#666';
    }
}

// Export for both CommonJS and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LOEAlgorithm;
}

// Also make available globally for browser usage
if (typeof window !== 'undefined') {
    window.LOEAlgorithm = LOEAlgorithm;
}

console.log('📊 LOE Algorithm module loaded successfully');
