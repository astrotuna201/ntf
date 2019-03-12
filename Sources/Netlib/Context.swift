//******************************************************************************
//  Created by Edward Connell on 3/12/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

public protocol Context {
    /// The current event log
    var log: Log { get set }


}