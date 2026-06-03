package algorithm

import "math"

// MixedRadix implements the Mixed-Radix FFT algorithm
// It factors a size n FFT into n1 * n2, computes several inner FFTs, then combines results
type MixedRadix struct {
	twiddles          []complex128
	widthFft          FftInterface
	width             int
	heightFft         FftInterface
	height            int
	length            int
	direction         Direction
	inplaceScratch    int
	outofplaceScratch int
}

// NewMixedRadix creates a MixedRadix FFT instance
// The FFT size will be widthFft.Len() * heightFft.Len()
func NewMixedRadix(widthFft, heightFft FftInterface) *MixedRadix {
	if widthFft.Direction() != heightFft.Direction() {
		panic("width and height FFTs must have the same direction")
	}

	direction := widthFft.Direction()
	width := widthFft.Len()
	height := heightFft.Len()
	length := width * height

	// Precompute twiddle factors
	twiddles := make([]complex128, length)
	for x := range width {
		for y := range height {
			idx := x*height + y
			angle := -2.0 * math.Pi * float64(x*y) / float64(length)
			if direction == Inverse {
				angle = -angle
			}
			twiddles[idx] = complex(math.Cos(angle), math.Sin(angle))
		}
	}

	// Scratch layout:
	// - first "length" values: intermediate matrix data (x-major)
	// - temp vector for gather/scatter of strided rows/columns
	// - scratch for inner FFT implementations
	heightInplace := heightFft.InplaceScratchLen()
	widthInplace := widthFft.InplaceScratchLen()
	tempLen := max(width, height)
	innerScratchLen := max(widthInplace, heightInplace)

	inplaceScratch := length + tempLen + innerScratchLen
	outofplaceScratch := inplaceScratch

	return &MixedRadix{
		twiddles:          twiddles,
		widthFft:          widthFft,
		width:             width,
		heightFft:         heightFft,
		height:            height,
		length:            length,
		direction:         direction,
		inplaceScratch:    inplaceScratch,
		outofplaceScratch: outofplaceScratch,
	}
}

func (m *MixedRadix) Len() int                  { return m.length }
func (m *MixedRadix) Direction() Direction      { return m.direction }
func (m *MixedRadix) InplaceScratchLen() int    { return m.inplaceScratch }
func (m *MixedRadix) OutOfPlaceScratchLen() int { return m.outofplaceScratch }
func (m *MixedRadix) ImmutableScratchLen() int  { return m.inplaceScratch }

func (m *MixedRadix) Process(buffer []complex128) {
	scratch := make([]complex128, m.InplaceScratchLen())
	m.ProcessWithScratch(buffer, scratch)
}

func (m *MixedRadix) ProcessWithScratch(buffer, scratch []complex128) {
	if len(scratch) < m.inplaceScratch {
		// Prevent a panic by allocating scratch if caller provided insufficient space
		scratch = make([]complex128, m.inplaceScratch)
	}

	// Intermediate storage in x-major layout: idx = x*height + y.
	selfScratch := scratch[:m.length]
	extra := scratch[m.length:]
	tempLen := max(m.width, m.height)
	temp := extra[:tempLen]
	innerScratch := extra[tempLen:]

	heightScratchNeed := m.heightFft.InplaceScratchLen()
	widthScratchNeed := m.widthFft.InplaceScratchLen()

	// Step 1: For each x, gather a strided column y from input (row-major), FFT(height), store contiguous in selfScratch.
	for x := 0; x < m.width; x++ {
		for y := 0; y < m.height; y++ {
			temp[y] = buffer[y*m.width+x]
		}

		if heightScratchNeed > 0 {
			m.heightFft.ProcessWithScratch(temp[:m.height], innerScratch[:heightScratchNeed])
		} else {
			m.heightFft.ProcessWithScratch(temp[:m.height], nil)
		}

		for y := 0; y < m.height; y++ {
			idx := x*m.height + y
			selfScratch[idx] = temp[y] * m.twiddles[idx]
		}
	}

	// Step 2: For each y, gather row across x, FFT(width), then store output in natural frequency order:
	// k = y + height*x => index x*height + y.
	for y := 0; y < m.height; y++ {
		for x := 0; x < m.width; x++ {
			temp[x] = selfScratch[x*m.height+y]
		}

		if widthScratchNeed > 0 {
			m.widthFft.ProcessWithScratch(temp[:m.width], innerScratch[:widthScratchNeed])
		} else {
			m.widthFft.ProcessWithScratch(temp[:m.width], nil)
		}

		for x := 0; x < m.width; x++ {
			buffer[x*m.height+y] = temp[x]
		}
	}
}

func (m *MixedRadix) ProcessOutOfPlace(input, output, scratch []complex128) {
	copy(output, input)
	m.ProcessWithScratch(output, scratch)
}

func (m *MixedRadix) ProcessImmutable(input []complex128, output, scratch []complex128) {
	copy(output, input)
	m.ProcessWithScratch(output, scratch)
}

// transpose performs a matrix transpose
// Treats input as a rows x cols matrix and transposes to output
func transpose(rows, cols int, input, output []complex128) {
	for r := range rows {
		for c := range cols {
			inputIdx := r*cols + c
			outputIdx := c*rows + r
			output[outputIdx] = input[inputIdx]
		}
	}
}
