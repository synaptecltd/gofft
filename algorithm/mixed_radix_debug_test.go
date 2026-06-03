package algorithm

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestMixedRadixSize6Debug - Detailed debugging for size 6 (2x3)
func TestMixedRadixSize6Debug(t *testing.T) {
	n1, n2 := 2, 3
	n := n1 * n2 // 6

	// Create simple input
	input := []complex128{
		complex(0, 0),
		complex(1, 0),
		complex(2, 0),
		complex(3, 0),
		complex(4, 0),
		complex(5, 0),
	}

	t.Logf("Input: %v", input)

	// Compute expected with DFT
	expected := make([]complex128, n)
	copy(expected, input)
	dft := NewDft(n, Forward)
	dft.ProcessWithScratch(expected, make([]complex128, n))
	t.Logf("Expected (DFT): %v", expected)

	// Compute with MixedRadix
	width := NewDft(n1, Forward)
	height := NewDft(n2, Forward)
	mr := NewMixedRadix(width, height)

	result := make([]complex128, n)
	copy(result, input)
	scratch := make([]complex128, mr.InplaceScratchLen())

	t.Logf("\n=== Starting MixedRadix ===")
	t.Logf("Width=%d, Height=%d", mr.width, mr.height)
	mr.ProcessWithScratch(result, scratch)
	t.Logf("Final result: %v", result)

	// Compare
	t.Logf("\n=== Comparison ===")
	maxErr := 0.0
	for i := range result {
		err := cmplx.Abs(result[i] - expected[i])
		if err > maxErr {
			maxErr = err
		}
		t.Logf("[%d] got=%v want=%v err=%.6e", i, result[i], expected[i], err)
	}

	t.Logf("\nMax error: %.6e", maxErr)

	if maxErr > 1e-10 {
		t.Errorf("Size 6 failed with error %.6e", maxErr)
	}
}

// TestMixedRadixManualSize6 - Manual implementation to verify algorithm
func TestMixedRadixManualSize6(t *testing.T) {
	// Size 6 = 2x3 using the same decomposition as MixedRadix:
	// 1) FFTs over y for each x (strided gather)
	// 2) twiddle multiply exp(-2pi*i*x*y/N)
	// 3) FFTs over x for each y

	input := []complex128{
		complex(0, 0),
		complex(1, 0),
		complex(2, 0),
		complex(3, 0),
		complex(4, 0),
		complex(5, 0),
	}

	t.Logf("Input: %v", input)

	width, height := 2, 3
	stage := make([]complex128, 6) // x-major: idx = x*height + y
	temp := make([]complex128, height)

	dft3 := NewDft(height, Forward)
	for x := range width {
		for y := range height {
			temp[y] = input[y*width+x]
		}
		t.Logf("Column x=%d before FFT3: %v", x, temp)
		dft3.ProcessWithScratch(temp, make([]complex128, height))
		t.Logf("Column x=%d after FFT3:  %v", x, temp)

		for y := range height {
			angle := -2.0 * math.Pi * float64(x*y) / 6.0
			tw := complex(math.Cos(angle), math.Sin(angle))
			stage[x*height+y] = temp[y] * tw
		}
	}
	t.Logf("After twiddles (x-major): %v", stage)

	result := make([]complex128, 6)
	temp2 := make([]complex128, width)
	dft2 := NewDft(width, Forward)
	for y := range height {
		for x := range width {
			temp2[x] = stage[x*height+y]
		}
		t.Logf("Row y=%d before FFT2: %v", y, temp2)
		dft2.ProcessWithScratch(temp2, make([]complex128, width))
		t.Logf("Row y=%d after FFT2:  %v", y, temp2)

		for x := range width {
			result[x*height+y] = temp2[x]
		}
	}
	t.Logf("Final result: %v", result)

	// Compare with DFT
	expected := make([]complex128, 6)
	copy(expected, input)
	dft := NewDft(6, Forward)
	dft.ProcessWithScratch(expected, make([]complex128, 6))

	t.Logf("\n=== Comparison ===")
	t.Logf("Expected: %v", expected)
	maxErr := 0.0
	for i := range result {
		err := cmplx.Abs(result[i] - expected[i])
		if err > maxErr {
			maxErr = err
		}
		t.Logf("[%d] got=%v want=%v err=%.6e", i, result[i], expected[i], err)
	}

	t.Logf("\nMax error: %.6e", maxErr)

	if maxErr > 1e-10 {
		t.Errorf("Manual size 6 failed with error %.6e", maxErr)
	}
}
