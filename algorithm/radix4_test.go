package algorithm

import (
	"math"
	"math/cmplx"
	"testing"
)

// Test Radix4 for size 32
func TestRadix4Size32(t *testing.T) {
	size := 32
	fft := NewRadix4(size, Forward)

	// Create simple test input
	input := make([]complex128, size)
	for i := range input {
		input[i] = complex(float64(i), 0)
	}

	// Compute FFT
	buffer := make([]complex128, size)
	copy(buffer, input)
	scratch := make([]complex128, fft.InplaceScratchLen())
	fft.ProcessWithScratch(buffer, scratch)

	// Compute expected using DFT
	expected := make([]complex128, size)
	dft := NewDft(size, Forward)
	dftScratch := make([]complex128, dft.InplaceScratchLen())
	copy(expected, input)
	dft.ProcessWithScratch(expected, dftScratch)

	// Compare
	maxError := 0.0
	for i := range buffer {
		err := cmplx.Abs(buffer[i] - expected[i])
		if err > maxError {
			maxError = err
		}
		if err > 1e-10 {
			t.Errorf("[%d] got %v, want %v, err %v", i, buffer[i], expected[i], err)
		}
	}

	if maxError > 1e-10 {
		t.Errorf("Max error: %v", maxError)
	}
}

// Test bit-reversed transpose
func TestBitReversedTranspose(t *testing.T) {
	// Test with dimensions valid for bitReversedTranspose4: width must be a power of 4.
	size := 32
	baseLen := 8 // width=4

	input := make([]complex128, size)
	for i := range input {
		input[i] = complex(float64(i), 0)
	}

	output := make([]complex128, size)
	bitReversedTranspose4(baseLen, input, output)

	// Print the transpose result to see the pattern
	t.Logf("Input:  %v", input[:8])
	t.Logf("Output: %v", output[:8])

	// The transpose should arrange data for radix-4.
	// For baseLen=8, we have 4 rows (size/baseLen = 4), bit-reversed in base-4 space.

	// Check a few values
	// Column 0 should have row 0 value (bit-reverse of 0 = 0)
	if output[0] != input[0] {
		t.Errorf("output[0] should be input[0], got %v", output[0])
	}
}

// Test twiddle factor computation
func TestTwiddleFactors(t *testing.T) {
	size := 32
	fft := NewRadix4(size, Forward)

	expectedCount := 0
	crossFftLen := fft.baseLen
	for crossFftLen < size {
		expectedCount += crossFftLen * 3
		crossFftLen *= 4
	}

	if len(fft.twiddles) != expectedCount {
		t.Errorf("Expected %d twiddles, got %d", expectedCount, len(fft.twiddles))
	}

	if expectedCount == 0 {
		return
	}

	// Check first few twiddle factors of the first stage.
	firstStageLen := fft.baseLen * 4
	for j := 1; j <= 3; j++ {
		angle := -2.0 * math.Pi * float64(0*j) / float64(firstStageLen)
		expected := complex(math.Cos(angle), math.Sin(angle))
		got := fft.twiddles[j-1]
		if cmplx.Abs(got-expected) > 1e-10 {
			t.Errorf("Twiddle[%d] got %v, want %v", j-1, got, expected)
		}
	}
}
