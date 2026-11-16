package com.example.facescorelab;
import android.graphics.Bitmap;
import org.json.JSONObject;

public final class RobustnessContractChecks {
    private RobustnessContractChecks(){}
    public static boolean verifyPartialFailureJson()throws Exception{
        Bitmap image=Bitmap.createBitmap(48,48,Bitmap.Config.ARGB_8888);final int[] calls={0};
        try{
            RobustnessProbe.Result result=RobustnessProbe.run(image,null,value->{if(calls[0]++==2)throw new Exception("synthetic failure");return 3.0;},null);
            JSONObject json=result.toJson();
            if(result.validCount!=8||!json.getJSONArray("samples").getJSONObject(2).isNull("value")||json.toString().contains("NaN"))throw new AssertionError("Failure must remain explicit and serializable");
            if(image.isRecycled())throw new AssertionError("Caller image was recycled");return true;
        }finally{if(!image.isRecycled())image.recycle();}
    }
    public static boolean verifyMissingBaselineIsNotFabricated()throws Exception{
        Bitmap image=Bitmap.createBitmap(48,48,Bitmap.Config.ARGB_8888);final int[] calls={0};
        try{
            RobustnessProbe.Result result=RobustnessProbe.run(image,null,value->{if(calls[0]++==0)throw new Exception("missing original");return 3.0;},null);
            JSONObject json=result.toJson();
            if(!json.isNull("baseline")||!json.isNull("maxAbsDelta")||!json.isNull("relativeDrift"))throw new AssertionError("A perturbed score was substituted for the original");
            return true;
        }finally{if(!image.isRecycled())image.recycle();}
    }
}
