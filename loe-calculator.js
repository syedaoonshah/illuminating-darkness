/**
 * LOE Calculator Library
 * Lightness Order Error - Image Enhancement Quality Assessment
 * 
 * This library provides functionality to calculate LOE (Lightness Order Error)
 * values using the exact MATLAB algorithm for image enhancement quality assessment.
 */

class LOECalculator {
    constructor() {
        // Initialize calculator
        console.log('🧮 LOE Calculator initialized');
    }

    /**
     * Calculate LOE between input and enhanced images
     * @param {ImageData} inputData - Canvas ImageData of input image
     * @param {ImageData} enhancedData - Canvas ImageData of enhanced image
     * @returns {Promise<number>} LOE value
     */
    async computeLOE(inputData, enhancedData) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                try {
                    const result = this.loeAlgorithm(inputData, enhancedData);
                    resolve(result);
                } catch (error) {
                    console.error('Error in LOE algorithm:', error);
                    reject(error);
                }
            }, 100);
        });
    }

    /**
     * Main LOE algorithm implementation
     * Translated from MATLAB code to JavaScript
     * @param {ImageData} inputData - Input image data
     * @param {ImageData} enhancedData - Enhanced image data
     * @returns {number} LOE value
     */
    loeAlgorithm(inputData, enhancedData) {
        const { width, height } = inputData;
        const m = height; // rows
        const n = width;  // columns
        
        // Convert to grayscale and get local maxima
        const win = 7;
        const inputGray = this.rgbToGray(inputData);
        const enhancedGray = this.rgbToGray(enhancedData);
        
        const imax = this.getLocalMax(inputGray, n, m, win);
        const emax = this.getLocalMax(enhancedGray, n, m, win);
        
        // Downsample
        const blkwin = 50;
        const mind = Math.min(m, n);
        const step = Math.max(1, Math.floor(mind / blkwin));
        const blkm = Math.floor(m / step);
        const blkn = Math.floor(n / step);
        
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
        
        return totalLOE / (blkm * blkn);
    }

    /**
     * Convert RGB image data to grayscale using max(R,G,B)
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
                gray[i][j] = Math.max(r, g, b); // Max of RGB channels as in MATLAB code
            }
        }
        return gray;
    }

    /**
     * Get local maximum values for each pixel
     * @param {Array<Array<number>>} image - 2D grayscale image array
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @param {number} win - Window size for local maximum
     * @returns {Array<Array<number>>} Local maximum values
     */
    getLocalMax(image, width, height, win) {
        const extImage = this.extendImage(image, width, height, win);
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
     * Extend image borders by reflection (equivalent to MATLAB's getextpic)
     * @param {Array<Array<number>>} image - 2D image array
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @param {number} winSize - Window size for extension
     * @returns {Array<Array<number>>} Extended image
     */
    extendImage(image, width, height, winSize) {
        const extended = [];
        const newHeight = height + 2 * winSize;
        const newWidth = width + 2 * winSize;
        
        // Initialize extended image with zeros
        for (let i = 0; i < newHeight; i++) {
            extended[i] = new Array(newWidth).fill(0);
        }
        
        // Copy original image to center
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                extended[i + winSize][j + winSize] = image[i][j];
            }
        }
        
        // Extend top and bottom edges by reflection
        for (let i = 0; i < winSize; i++) {
            for (let j = winSize; j < winSize + width; j++) {
                extended[winSize - 1 - i][j] = extended[winSize + 1 + i][j]; // top edge
                extended[height + winSize + i][j] = extended[height + winSize - 2 - i][j]; // bottom edge
            }
        }
        
        // Extend left and right edges by reflection
        for (let i = 0; i < newHeight; i++) {
            for (let j = 0; j < winSize; j++) {
                extended[i][winSize - 1 - j] = extended[i][winSize + 1 + j]; // left edge
                extended[i][width + winSize + j] = extended[i][width + winSize - 2 - j]; // right edge
            }
        }
        
        return extended;
    }

    /**
     * Convert Image object to ImageData
     * @param {Image} img - HTML Image object
     * @param {number} targetWidth - Target width (optional)
     * @param {number} targetHeight - Target height (optional)
     * @returns {ImageData} Canvas ImageData
     */
    getImageData(img, targetWidth = null, targetHeight = null) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        const width = targetWidth || img.width;
        const height = targetHeight || img.height;
        
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);
        return ctx.getImageData(0, 0, width, height);
    }

    /**
     * Calculate LOE between two Image objects
     * @param {Image} inputImage - Input image
     * @param {Image} enhancedImage - Enhanced image
     * @returns {Promise<number>} LOE value
     */
    async calculateImageLOE(inputImage, enhancedImage) {
        // Ensure both images have the same dimensions (use input as reference)
        const targetWidth = inputImage.width;
        const targetHeight = inputImage.height;
        
        const inputData = this.getImageData(inputImage, targetWidth, targetHeight);
        const enhancedData = this.getImageData(enhancedImage, targetWidth, targetHeight);
        
        return await this.computeLOE(inputData, enhancedData);
    }

    /**
     * Get performance rating based on LOE value
     * @param {number} loeValue - LOE score
     * @returns {string} Performance rating
     */
    getLOEPerformanceRating(loeValue) {
        if (loeValue < 35) return 'Excellent';
        if (loeValue < 50) return 'Good';
        if (loeValue < 65) return 'Fair';
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

    /**
     * Get interpretation text for LOE value
     * @param {number} loeValue - LOE score
     * @returns {string} Interpretation text
     */
    getInterpretation(loeValue) {
        if (loeValue < 50) {
            return 'Excellent enhancement quality - very good preservation of lightness order';
        } else if (loeValue < 100) {
            return 'Good enhancement quality - acceptable lightness order preservation';
        } else if (loeValue < 200) {
            return 'Fair enhancement quality - some lightness order distortion';
        } else {
            return 'Poor enhancement quality - significant lightness order distortion';
        }
    }

    /**
     * Load image from URL with error handling
     * @param {string} imageUrl - Image URL
     * @returns {Promise<Image|null>} Loaded image or null if failed
     */
    async loadImageFromPath(imageUrl) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = 'anonymous'; // Enable CORS
            
            img.onload = () => {
                resolve(img);
            };
            
            img.onerror = () => {
                resolve(null); // Return null instead of rejecting
            };
            
            img.src = imageUrl;
            
            // Set a timeout to avoid hanging
            setTimeout(() => {
                if (!img.complete) {
                    resolve(null);
                }
            }, 5000); // 5 second timeout
        });
    }

    /**
     * Create a delay for async processing
     * @param {number} ms - Delay in milliseconds
     * @returns {Promise} Promise that resolves after delay
     */
    simulateDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LOECalculator;
} else if (typeof window !== 'undefined') {
    window.LOECalculator = LOECalculator;
}
