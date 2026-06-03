package algorithm

import (
	"fmt"
	"testing"
)

// BenchmarkRadersVsBluestein compares Rader's and Bluestein's performance
func BenchmarkRadersVsBluestein(b *testing.B) {
	primes := []int{11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}

	for _, p := range primes {
		// Benchmark Rader's
		b.Run(fmt.Sprintf("Raders/Size %d", p), func(b *testing.B) {
			innerFft := NewDft(p-1, Forward)
			raders := NewRaders(innerFft)
			buffer := make([]complex128, p)
			scratch := make([]complex128, raders.InplaceScratchLen())

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				raders.ProcessWithScratch(buffer, scratch)
			}
		})

		// Benchmark Bluestein's
		b.Run(fmt.Sprintf("Bluestein/Size %d", p), func(b *testing.B) {
			bluestein := NewBluestein(p, Forward)
			buffer := make([]complex128, p)
			scratch := make([]complex128, bluestein.InplaceScratchLen())

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				bluestein.ProcessWithScratch(buffer, scratch)
			}
		})
	}
}

// BenchmarkRadersWithOptimizedInner tests Rader's with optimized inner FFTs
func BenchmarkRadersWithOptimizedInner(b *testing.B) {
	testCases := []struct {
		prime int
		desc  string
	}{
		{17, "Prime17(innerFFT=16=Butterfly16)"},
		{37, "Prime37(innerFFT=36=6x6)"},
		{257, "Prime257(innerFFT=256=Radix4)"},
	}

	for _, tc := range testCases {
		b.Run(tc.desc, func(b *testing.B) {
			if !isPrime(tc.prime) {
				b.Skipf("skipping non-prime size %d for Rader benchmark", tc.prime)
			}

			var innerFft FftInterface

			innerLen := tc.prime - 1
			// Use appropriate optimized FFT for inner
			switch innerLen {
			case 16:
				innerFft = NewButterfly16(Forward)
			case 36:
				innerFft = NewMixedRadix(NewButterfly6(Forward), NewButterfly6(Forward))
			case 256:
				innerFft = NewRadix4(innerLen, Forward)
			default:
				if isPowerOfTwo(innerLen) {
					if innerLen <= 32 {
						switch innerLen {
						case 32:
							innerFft = NewButterfly32(Forward)
						default:
							innerFft = NewDft(innerLen, Forward)
						}
					} else {
						innerFft = NewRadix4(innerLen, Forward)
					}
				} else {
					innerFft = NewDft(innerLen, Forward)
				}
			}

			raders := NewRaders(innerFft)
			buffer := make([]complex128, tc.prime)
			scratch := make([]complex128, raders.InplaceScratchLen())

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				raders.ProcessWithScratch(buffer, scratch)
			}
		})
	}
}
