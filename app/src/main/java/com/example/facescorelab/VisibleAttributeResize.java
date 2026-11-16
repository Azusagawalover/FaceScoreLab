package com.example.facescorelab;

/** Separable RGB bilinear resizing with antialiasing and per-axis 8-bit rounding.
 * This optional-model path matches Pillow 13.0.0 BILINEAR preparation used in training.
 * Algorithm reference: Pillow 13.0.0 src/libImaging/Resample.c.
 * Attribution/license: project THIRD_PARTY_VISIBLE_RESIZE_LICENSE.txt.
 * Keep the original beauty-model preprocessing separate.
 */
final class VisibleAttributeResize {
    private static final int PRECISION_BITS=22;
    private VisibleAttributeResize(){}

    static int[] resize(int[] pixels,int width,int height,int targetWidth,int targetHeight){
        if(width<1||height<1||targetWidth<1||targetHeight<1||pixels.length!=width*height)
            throw new IllegalArgumentException("Invalid RGB resize dimensions");
        // Pillow 13 chooses vertical first when it removes over twice as many rows as columns.
        boolean horizontalFirst=!((height-targetHeight)>0&&(height-targetHeight)>(width-targetWidth)*2L);
        if(horizontalFirst)return vertical(horizontal(pixels,width,height,targetWidth),targetWidth,height,targetHeight);
        return horizontal(vertical(pixels,width,height,targetHeight),width,targetHeight,targetWidth);
    }
    private static int[] horizontal(int[] pixels,int width,int height,int targetWidth){
        if(width==targetWidth)return pixels;
        Coefficients horizontal=new Coefficients(width,targetWidth);
        int[] intermediate=new int[targetWidth*height];
        for(int y=0;y<height;y++)for(int x=0;x<targetWidth;x++){
            int r=1<<(PRECISION_BITS-1),g=r,b=r;
            for(int k=0;k<horizontal.weights[x].length;k++){
                int pixel=pixels[y*width+horizontal.starts[x]+k],weight=horizontal.weights[x][k];
                r+=((pixel>>16)&255)*weight;g+=((pixel>>8)&255)*weight;b+=(pixel&255)*weight;
            }
            intermediate[y*targetWidth+x]=0xff000000|(clip(r)<<16)|(clip(g)<<8)|clip(b);
        }
        return intermediate;
    }
    private static int[] vertical(int[] pixels,int width,int height,int targetHeight){
        if(height==targetHeight)return pixels;
        Coefficients vertical=new Coefficients(height,targetHeight);
        int[] result=new int[width*targetHeight];
        for(int y=0;y<targetHeight;y++)for(int x=0;x<width;x++){
            int r=1<<(PRECISION_BITS-1),g=r,b=r;
            for(int k=0;k<vertical.weights[y].length;k++){
                int pixel=pixels[(vertical.starts[y]+k)*width+x],weight=vertical.weights[y][k];
                r+=((pixel>>16)&255)*weight;g+=((pixel>>8)&255)*weight;b+=(pixel&255)*weight;
            }
            result[y*width+x]=0xff000000|(clip(r)<<16)|(clip(g)<<8)|clip(b);
        }
        return result;
    }
    static float[] imageNetTensor(int[] pixels){
        int area=pixels.length;float[] input=new float[area*3];
        for(int i=0;i<area;i++){
            int pixel=pixels[i];input[i]=(((pixel>>16)&255)/255f-.485f)/.229f;
            input[area+i]=(((pixel>>8)&255)/255f-.456f)/.224f;input[2*area+i]=((pixel&255)/255f-.406f)/.225f;
        }
        return input;
    }
    private static int clip(int value){return Math.max(0,Math.min(255,value>>PRECISION_BITS));}
    private static final class Coefficients {
        final int[] starts;
        final int[][] weights;
        Coefficients(int source,int target){
            starts=new int[target];weights=new int[target][];
            double scale=source/(double)target,filterScale=Math.max(1,scale),support=filterScale,inverseFilterScale=1/filterScale;
            for(int output=0;output<target;output++){
                double center=(output+.5)*scale;
                int first=Math.max(0,(int)(center-support+.5));
                int last=Math.min(source,(int)(center+support+.5));
                starts[output]=first;weights[output]=new int[last-first];
                double[] raw=new double[last-first];double total=0;
                for(int k=0;k<raw.length;k++){
                    raw[k]=Math.max(0,1-Math.abs((k+first-center+.5)*inverseFilterScale));total+=raw[k];
                }
                for(int k=0;k<raw.length;k++)weights[output][k]=(int)(.5+raw[k]/total*(1<<PRECISION_BITS));
            }
        }
    }
}
