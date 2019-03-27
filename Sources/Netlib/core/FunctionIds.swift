//******************************************************************************
//  Created by Edward Connell on  3/25/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

public struct FunctionId {
    public static let Abs = UUID(uuidString: "B0ACCA29-21B3-4D54-B59D-1E85755D7687")!
    public static let Add = UUID(uuidString: "2BDCDAB2-6C9A-4150-9F6D-AA87725CEB8C")!
    public static let AndAll = UUID(uuidString: "1226E367-75A1-415F-BE97-854C89E01EB4")!
    public static let ApproximateEqual = UUID(uuidString: "BCA656F1-E504-4089-8317-FDAC382D9C3A")!
    public static let ArgMax = UUID(uuidString: "F1CE2CD5-790F-4E3E-A478-4066DE35BF2A")!
    public static let ArgMin = UUID(uuidString: "52125F2E-EFB5-4C54-BF2B-9094D5D4495F")!
    public static let Broadcast = UUID(uuidString: "1C05B839-8E5C-4125-9376-0F5B65AF4B3F")!
    public static let Cast = UUID(uuidString: "9CFD6BDE-750B-47DE-AE50-571E0AAF412D")!
    public static let Ceil = UUID(uuidString: "F25BA913-59BC-46FA-95EB-C09E9187BB93")!
    public static let Conv2D = UUID(uuidString: "A055F634-E76E-4B66-AA45-BACF5C9F0247")!
    public static let Concatenate = UUID(uuidString: "BBD13F37-8B66-47DA-81C6-AE70F9C5E0DB")!
    public static let Cos = UUID(uuidString: "094D6AA5-C77F-4433-AB63-35B90934770C")!
    public static let Cosh = UUID(uuidString: "7FFF81D5-28CC-40FA-B510-8ECD02BCD73D")!
    public static let Div = UUID(uuidString: "A9FEE793-84E5-4B79-8B2E-8D25DEC5789D")!
    public static let Equal = UUID(uuidString: "4890A4FD-382F-4467-BCAD-68B6A5A428C8")!
    public static let Exp = UUID(uuidString: "366D980E-0C9B-4E97-ABC6-5005F606FF7C")!
    public static let Floor = UUID(uuidString: "5B1B1F13-1ED0-47ED-8FDB-E79A1F808B89")!
    public static let Greater = UUID(uuidString: "9A39BD5D-D14E-43A5-B3D6-10A5CB96F543")!
    public static let GreaterEqual = UUID(uuidString: "C6E6119A-DCB4-423C-BABB-C65010FDD3E9")!
    public static let Less = UUID(uuidString: "AEFA893C-E268-4B10-8A3F-CFCF0C3BA7BF")!
    public static let LessEqual = UUID(uuidString: "68DE346E-84B0-411E-A138-D461E7E7CDB0")!
    public static let Log = UUID(uuidString: "9470A3C9-68F4-4116-85FC-C3FBC89F2471")!
    public static let LogicalNot = UUID(uuidString: "B2C5F168-EC4C-41C4-8C52-F1EB345EAEEE")!
    public static let LogicalAnd = UUID(uuidString: "B87350E1-D56F-4EC7-8511-6E1C4208D74F")!
    public static let LogicalOr = UUID(uuidString: "302A9486-20EA-4DC1-A842-78984354F73B")!
    public static let LogSoftmax = UUID(uuidString: "93884FAC-B022-4DA5-9B36-49AD3AD01028")!
    public static let MatMul = UUID(uuidString: "63856834-ABF5-4285-8F4A-6972BB8515BD")!
    public static let Max = UUID(uuidString: "C1DD0D5D-E234-42F0-80D4-6D55044888A0")!
    public static let Maximum = UUID(uuidString: "294F50D8-9BA1-4FC3-9EE0-074E67E5DE6E")!
    public static let Mean = UUID(uuidString: "70DA84E4-C5FD-43C6-9C2E-3BAA2712EF02")!
    public static let Min = UUID(uuidString: "A50EB375-AB89-44D2-9BA5-C2B083B2D292")!
    public static let Minimum = UUID(uuidString: "6E7D96D7-BB04-4298-92B7-1A401BB4F85A")!
    public static let Mod = UUID(uuidString: "50A75695-B8B7-4DD8-BADE-1CFA953542C1")!
    public static let Mul = UUID(uuidString: "EC595FB6-8CAE-480A-BF4F-D6AD9EB78F32")!
    public static let Neg = UUID(uuidString: "34918122-5A2D-420E-9A20-5FD98858A069")!
    public static let NotEqual = UUID(uuidString: "E663F846-27A2-4F41-AD95-AC938C71DD3D")!
    public static let OrAny = UUID(uuidString: "11486E0D-DA33-4A94-8D76-8E75B9353E9D")!
    public static let Pad = UUID(uuidString: "991222DC-9621-4FD1-B52A-91B507F53B07")!
    public static let Pow = UUID(uuidString: "3ECBF14B-A89E-4DB7-9CED-577344582F0E")!
    public static let Prod = UUID(uuidString: "500A9345-8EC6-4D0A-9C21-C0500091C2A7")!
    public static let Rsqrt = UUID(uuidString: "E571EB5B-B94E-4A7C-A769-2278A3339E42")!
    public static let Select = UUID(uuidString: "40759C84-5A65-4214-B073-A1B14C43822A")!
    public static let Sin = UUID(uuidString: "D7FD3DCF-7487-4D58-A1CC-0FB92A29F03A")!
    public static let Sinh = UUID(uuidString: "94ABC1F0-B17E-4AA2-8A75-A64A96B0F0DE")!
    public static let Square = UUID(uuidString: "FDDE418D-734A-485F-B18B-ABEB3DA23E4D")!
    public static let SquaredDifference = UUID(uuidString: "0C85D538-7821-4009-8C01-5B70B691B922")!
    public static let Sqrt = UUID(uuidString: "1C80969D-535F-4719-9D01-E8F336D2A1D8")!
    public static let Subtract = UUID(uuidString: "164A3A5E-7502-416E-87BC-6CEFB7EB2028")!
    public static let Sum = UUID(uuidString: "B8C9777B-8E7D-4A23-9BD9-9F9E659483AB")!
    public static let Tan = UUID(uuidString: "506EEF1C-255A-4BCD-A3B0-BB1F04E31EC5")!
    public static let Tanh = UUID(uuidString: "D66ECA40-0F52-4D7A-9A14-686A78875D15")!
}
