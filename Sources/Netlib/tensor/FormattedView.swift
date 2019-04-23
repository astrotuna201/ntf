//******************************************************************************
//  Created by Edward Connell on 4/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorView default implementation
public extension TensorView where Scalar: AnyConvertable {
    //--------------------------------------------------------------------------
    // formatted
    func formatted(
        maxCols: Int = 10,
        maxItems: [Int]? = nil,
        numberFormat: (width: Int, precision: Int)? = nil) -> String {
        
        guard !shape.isEmpty else { return "[Empty]\n" }
        var string = ""
        var index = [Int](repeating: 0, count: shape.rank)
        var iterator = self.values().makeIterator()
        var itemCount = 0
        let indentSize = "  "
        let extents = shape.padded(with: padding).extents
        let lastDimension = shape.lastDimension

        // clamp ranges
        let maxItems = maxItems?.enumerated().map {
            min($1, extents[$0])
        } ?? extents

        // set header
        string += "\nTensorView extents: \(shape.extents.description)" +
        " paddedExtents: \(extents.description)\n"

        func appendFormatted(value: Scalar) {
            if let fmt = numberFormat {
                let str = String(format: Scalar.formatString(fmt),
                                 value.asCVarArg)
                string += "\(str) "
            } else {
                string += "\(value.asString) "
            }
        }

        // recursive rank > 1 formatting
        func format(dim: Int, indent: String) {
            // print the heading unless it's the last two which we print
            // 2d matrix style
            if dim == lastDimension - 1 {
                let header = "at index: \(String(describing: index))"
                string += "\(indent)\(header)\n\(indent)"
                string += String(repeating: "-", count: header.count) + "\n"
                let maxCol = extents[lastDimension] - 1
                let lastCol = maxItems[lastDimension] - 1

                for _ in 0..<maxItems[lastDimension - 1] {
                    string += indent
                    for col in 0...lastCol {
                        if let value = iterator.next() {
                            appendFormatted(value: value)
                            if col == lastCol {
                                string += (col < maxCol) ? " ...\n" : "\n"
                            }
                        }
                    }
                }
                string += "\n\n"

            } else {
                for _ in 0..<maxItems[dim] {
                    // output index header
                    let header = indent +
                    "at index: \(String(describing: index))"
                    string += "\(indent)\(header)\n\(indent)"
                    string += String(repeating: "=", count: header.count) + "\n"

                    // recursively call next contained dimension
                    format(dim: dim + 1, indent: indent + indentSize)
                    index[dim] += 1
                }
            }
        }

        // format based on rank
        switch shape.rank {
        case 0, 1:
            if shape.isScalar {
                let value = iterator.next()!
                appendFormatted(value: value)
                string += "\n"
            } else {
                var col = 0
                while let value = iterator.next(), itemCount < maxItems[0] {
                    appendFormatted(value: value)
                    itemCount += 1
                    col += 1
                    if col == maxCols {
                        string += "\n"
                        col = 0
                    }
                }
            }
            string += "\n"

        default:
            format(dim: 0, indent: "")
            string = String(string.dropLast())
        }

        return string
    }
}
