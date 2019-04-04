//******************************************************************************
//  Created by Edward Connell on 4/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    // helper to flip highlight color back and forth
    // it doesn't work in the Xcode console for some reason,
    // but it's nice when using CLion
    func setStringColor(text: inout String, highlight: Bool,
                        currentColor: inout LogColor,
                        normalColor: LogColor = .white,
                        highlightColor: LogColor = .blue) {
        #if os(Linux)
        if currentColor == normalColor && highlight {
            text += highlightColor.rawValue
            currentColor = highlightColor
            
        } else if currentColor == highlightColor && !highlight {
            text += normalColor.rawValue
            currentColor = normalColor
        }
        #endif
    }
}

public extension VectorTensorView where Scalar: AnyConvertable {
    //    //--------------------------------------------------------------------------
    //    // formatted
    //    func formatted(
    //        precision: Int = 6,
    //        columnWidth: Int = 9,
    //        maxCols: Int = 10,
    //        maxItems: Int = Int.max,
    //        highlightThreshold: Float = Float.greatestFiniteMagnitude) -> String {
    //
    //        // setup
    //        let itemCount = min(shape.items, maxItems)
    //        let itemStride = shape.strides[0]
    //        var string = "DataView extent \(shape.extents.description)\n"
    //        var currentColor = LogColor.white
    //        string += currentColor.rawValue
    //
    //        do {
    //            let buffer = try ro()
    //            string = "DataView extent [\(shape.items)]\n"
    //            let pad = itemCount > 9 ? " " : ""
    //
    //            for item in 0..<itemCount {
    //                if item < 10 { string += pad }
    //                string += "[\(item)] "
    //
    //                let value = buffer[item * itemStride]
    //                setStringColor(text: &string,
    //                               highlight: value.asFloat > highlightThreshold,
    //                               currentColor: &currentColor)
    //                string += "\(String(format: format, value.asCVarArg))\n"
    //            }
    //            string += "\n"
    //        } catch {
    //            string += String(describing: error)
    //        }
    //        return string
    //    }
}
