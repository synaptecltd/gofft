package algorithm

import "math"

// RadixFactor represents a small radix (2-7) used in factorization
type RadixFactor int

const (
	Factor2 RadixFactor = 2
	Factor3 RadixFactor = 3
	Factor4 RadixFactor = 4
	Factor5 RadixFactor = 5
	Factor6 RadixFactor = 6
	Factor7 RadixFactor = 7
)

// TransposeFactor tracks a factor and how many times it appears consecutively
type TransposeFactor struct {
	factor RadixFactor
	count  int
}

// RadixN implements multi-factor FFT decomposition
// For sizes like 24 = 2³×3, 60 = 2²×3×5, 120 = 2³×3×5
type RadixN struct {
	length           int
	direction        Direction
	baseFft          FftInterface
	baseLen          int
	factors          []RadixFactor     // Original factors
	transposeFactors []TransposeFactor // Collapsed for transpose
	butterflies      []FftInterface    // Butterfly for each factor
	twiddles         []complex128      // All twiddle factors
	inplaceScratch   int
	crossFftChunk    []complex128 // Reusable chunk buffer (max radix 7)
}

// NewRadixN creates a RadixN FFT instance
// factors: array of radices (2-7) to decompose
// baseFft: FFT to use for base (often size 1)
func NewRadixN(factors []RadixFactor, baseFft FftInterface) *RadixN {
	baseLen := baseFft.Len()
	direction := baseFft.Direction()

	// Create butterflies for each factor
	butterflies := make([]FftInterface, len(factors))
	crossFftLen := baseLen
	twiddleCount := 0

	for i, factor := range factors {
		crossFftRows := int(factor)
		crossFftColumns := crossFftLen

		// Twiddles needed: columns × (rows - 1)
		twiddleCount += crossFftColumns * (crossFftRows - 1)

		// Create butterfly for this factor
		switch factor {
		case Factor2:
			butterflies[i] = NewButterfly2(direction)
		case Factor3:
			butterflies[i] = NewButterfly3(direction)
		case Factor4:
			butterflies[i] = NewButterfly4(direction)
		case Factor5:
			butterflies[i] = NewButterfly5(direction)
		case Factor6:
			butterflies[i] = NewButterfly6(direction)
		case Factor7:
			butterflies[i] = NewButterfly7(direction)
		default:
			panic("unsupported radix factor")
		}

		crossFftLen *= crossFftRows
	}

	length := crossFftLen

	// Build transpose factors (reversed and collapsed)
	transposeFactors := make([]TransposeFactor, 0, len(factors))
	for i := len(factors) - 1; i >= 0; i-- {
		f := factors[i]

		// Try to collapse with last factor
		if len(transposeFactors) > 0 && transposeFactors[len(transposeFactors)-1].factor == f {
			transposeFactors[len(transposeFactors)-1].count++
		} else {
			transposeFactors = append(transposeFactors, TransposeFactor{factor: f, count: 1})
		}
	}

	// Precompute all twiddle factors
	twiddles := make([]complex128, twiddleCount)
	twiddleIdx := 0
	crossFftLen = baseLen

	for _, factor := range factors {
		crossFftColumns := crossFftLen
		crossFftLen *= int(factor)

		// Twiddles for this layer
		for i := range crossFftColumns {
			for k := 1; k < int(factor); k++ {
				angle := -2.0 * math.Pi * float64(i*k) / float64(crossFftLen)
				if direction == Inverse {
					angle = -angle
				}
				twiddles[twiddleIdx] = complex(math.Cos(angle), math.Sin(angle))
				twiddleIdx++
			}
		}
	}

	// Calculate scratch space needed for base FFT and butterflies
	baseScratch := baseFft.InplaceScratchLen()
	maxButterflyScratch := 0
	for _, b := range butterflies {
		if s := b.InplaceScratchLen(); s > maxButterflyScratch {
			maxButterflyScratch = s
		}
	}

	inplaceScratch := length
	if baseScratch > length {
		inplaceScratch = length + baseScratch
	}
	if maxButterflyScratch > inplaceScratch-length {
		inplaceScratch = length + maxButterflyScratch
	}

	return &RadixN{
		length:           length,
		direction:        direction,
		baseFft:          baseFft,
		baseLen:          baseLen,
		factors:          factors,
		transposeFactors: transposeFactors,
		butterflies:      butterflies,
		twiddles:         twiddles,
		inplaceScratch:   inplaceScratch,
		crossFftChunk:    make([]complex128, 7), // Max radix is 7
	}
}

func (r *RadixN) Len() int               { return r.length }
func (r *RadixN) Direction() Direction   { return r.direction }
func (r *RadixN) InplaceScratchLen() int { return r.inplaceScratch }

func (r *RadixN) ProcessWithScratch(buffer, scratch []complex128) {
	// Process each chunk
	for i := 0; i < len(buffer); i += r.length {
		chunk := buffer[i : i+r.length]
		workScratch := scratch[:r.inplaceScratch]
		r.processOne(chunk, workScratch)
	}
}

func (r *RadixN) processOne(buffer, scratch []complex128) {
	output := scratch[:r.length]
	innerScratch := make([]complex128, r.baseFft.InplaceScratchLen())
	if len(scratch) > r.length {
		innerScratch = scratch[r.length:]
	}

	// Step 1: Factor transpose (reorders data based on factors)
	factorTranspose(r.baseLen, buffer, output, r.transposeFactors)

	// Step 2: Base FFTs
	r.baseFft.ProcessWithScratch(output, innerScratch)

	// Step 3: Cross-FFTs with twiddles for each factor layer
	crossFftLen := r.baseLen
	twiddleOffset := 0

	// Use pre-allocated buffers for cross-FFT (zero allocations)
	crossFftScratch := scratch[r.length:r.InplaceScratchLen()] // Use provided scratch

	for i, butterfly := range r.butterflies {
		radix := int(r.factors[i])
		crossFftColumns := crossFftLen
		crossFftLen *= radix

		// Apply cross-FFT butterflies on chunks
		layerTwiddles := r.twiddles[twiddleOffset : twiddleOffset+crossFftColumns*(radix-1)]

		// Use pre-allocated scratch with proper length
		butterflyRequiredLen := butterfly.InplaceScratchLen()
		butterflyScratch := crossFftScratch[:butterflyRequiredLen]

		for chunkStart := 0; chunkStart < r.length; chunkStart += crossFftLen {
			chunk := output[chunkStart : chunkStart+crossFftLen]
			applyCrossFft(chunk, layerTwiddles, crossFftColumns, radix, butterfly, r.crossFftChunk[:radix], butterflyScratch)
		}

		twiddleOffset += crossFftColumns * (radix - 1)
	}

	// Copy result back to buffer
	copy(buffer, output)
}

// factorTranspose performs a transpose with remainder-reversal on column indices
// This is like bit-reversal but generalized to mixed radices
func factorTranspose(height int, input, output []complex128, factors []TransposeFactor) {
	width := len(input) / height

	// Simple transpose with remainder reversal
	for x := range width {
		xRev := reverseRemainders(x, factors)
		for y := range height {
			inputIdx := x + y*width
			outputIdx := y + xRev*height
			output[outputIdx] = input[inputIdx]
		}
	}
}

// reverseRemainders performs remainder reversal (generalized bit reversal)
// Divides value by factors and builds result from remainders in reverse order
func reverseRemainders(value int, factors []TransposeFactor) int {
	result := 0

	for _, f := range factors {
		radix := int(f.factor)
		for i := 0; i < f.count; i++ {
			result = result*radix + (value % radix)
			value = value / radix
		}
	}

	return result
}

// applyCrossFft applies a cross-FFT butterfly with twiddles
// This performs radix-point butterflies on strided data
// Reuses pre-allocated chunk and scratch buffers to avoid per-iteration allocations.
func applyCrossFft(data []complex128, twiddles []complex128, columns, radix int, butterfly FftInterface, chunk, scratch []complex128) {
	// For each column
	for col := range columns {
		// First element (no twiddle)
		chunk[0] = data[col]

		// Remaining elements with twiddles
		// Twiddles are laid out: [col0_tw1, col0_tw2, ..., col1_tw1, col1_tw2, ...]
		for r := 1; r < radix; r++ {
			idx := col + r*columns
			twiddleIdx := col*(radix-1) + (r - 1)
			chunk[r] = data[idx] * twiddles[twiddleIdx]
		}

		// Apply butterfly
		butterfly.ProcessWithScratch(chunk, scratch)

		// Write back
		for r := range radix {
			idx := col + r*columns
			data[idx] = chunk[r]
		}
	}
}
