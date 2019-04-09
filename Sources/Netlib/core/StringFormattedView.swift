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
    func formatted(maxCols: Int = 10,
                   maxItems: Int = Int.max,
                   formatString: String? = nil) throws -> String {
        guard !shape.isEmpty else { return "[Empty]\n" }
        let formatStr = formatString ?? Scalar.defaultFormatString
        var string = ""
        var index = [Int](repeating: 0, count: shape.rank)
        var iterator = try self.values().makeIterator()
        var itemCount = 0
        let indentSize = "  "
        
        // set header
        string +=
        """
        TensorView extents: \(shape.extents.description)
             paddedExtents: \(shape.paddedExtents.description)
        """
        string += "\n"
        
        // recursive rank > 1 formatting
        func format(dim: Int, indent: String) {
            guard itemCount <= maxItems else { return }
            // print the heading
            if dim != shape.lastDimension {
                string += "\(indent)at index: \(String(describing: index))\n"
                index[dim] += 1
                format(dim: dim + 1, indent: indent + indentSize)
            } else {
                // format a multi line vector
                if shape.rank == 1 {
                    
                } else {
                    string += indent
                    // print columns
                    for col in 0..<shape.paddedExtents[dim] {
                        if let value = iterator.next() {
                            itemCount += 1
                            if col < maxCols {
                                let svalue = String(format: formatStr,
                                                    value.asCVarArg)
                                string += "\(svalue),"
                            }
                        } else {
                            return
                        }
                    }
                }
            }
        }

        // format based on rank
        switch shape.rank {
        case 1:
            if shape.isScalar {
                let value = iterator.next()!
                string += "\(String(format: formatStr, value.asCVarArg))\n\n"
            } else {
            }
        default:
            format(dim: 0, indent: "")
        }
        
        return string
    }
}
