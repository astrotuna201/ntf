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
        var string = ""
        var index = [Int](repeating: 0, count: shape.rank)
        var iterator = try self.values().makeIterator()
        var itemCount = 0
        let indentSize = "  "
        
        // set header
        string += "TensorView extents: \(shape.extents.description)" +
        " paddedExtents: \(shape.paddedExtents.description)\n"
        
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
                    let lastCol = min(shape.paddedExtents[dim], maxCols)
                    var col = 0
                    while let value = iterator.next(), itemCount < maxItems {
                        if let fmt = formatString {
                            string += "\(String(format: fmt, value.asCVarArg)), "
                        } else {
                            string += "\(value.asString), "
                        }
                        itemCount += 1
                        col += 1
                        if col == lastCol {
                            string += col < shape.paddedExtents[dim] ?
                                " ...\n\(indent)" : "\n\(indent)"
                            col = 0
                        }
                    }
                    string = String(string[..<string.lastIndex(of: ",")!])
                }
            }
        }

        // format based on rank
        switch shape.rank {
        case 1:
            if shape.isScalar {
                let value = iterator.next()!
                if let fmt = formatString {
                    string += "\(String(format: fmt, value.asCVarArg))\n"
                } else {
                    string += "\(value.asString)\n"
                }
            } else {
                var col = 0
                while let value = iterator.next(), itemCount < maxItems {
                    if let fmt = formatString {
                        string += "\(String(format: fmt, value.asCVarArg)), "
                    } else {
                        string += "\(value.asString), "
                    }
                    itemCount += 1
                    col += 1
                    if col == maxCols {
                        string += "\n"
                        col = 0
                    }
                }
                string = String(string[..<string.lastIndex(of: ",")!])
            }
            
        default:
            format(dim: 0, indent: "")
        }
        
        return string + "\n"
    }
}
