package com.example.facescorelab;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.graphics.drawable.StateListDrawable;
import android.view.Gravity;
import android.widget.LinearLayout;
import androidx.appcompat.widget.AppCompatButton;

/** 页面颜色、字号和控件样式。 */
public final class VisualStyle {
    /** 正文与标题。 */
    public static final int INK=0xff1c2431, MUTED=0xff625f59;
    /** 主色：深海藏蓝，用于可交互元素。 */
    public static final int ACCENT=0xff16324f, ACCENT_DEEP=0xff0c1f33, ACCENT_SOFT=0xffe7edf4;
    /** 点缀色：珊瑚橙（图形与大字号可用，小字请用 CORAL_DEEP）。 */
    public static final int CORAL=0xffd9562f, CORAL_DEEP=0xffc24a28, CORAL_SOFT=0xfffbe9e2;
    /** 数据与技术状态色。 */
    public static final int TEAL=0xff14766b, TEAL_SOFT=0xffe3f1ee;
    public static final int OK=0xff14766b, WARN=0xffa9661b, WARN_SOFT=0xfffaf0e1, DANGER=0xffa8322a;
    /** 容器与分隔。 */
    public static final int SOFT=0xffe7edf4, BORDER=0xffe4ded3, SURFACE=0xfff0eee8;
    public static final int CARD=0xffffffff, CARD_TINT=0xfff7f5f0;
    /** 禁用态前景：纸底对比度足够。 */
    public static final int DISABLED=0xff6f6b64;

    /** 大号数字（分数）。 */
    public static final float DISPLAY=40f;
    /** 页面主标题。 */
    public static final float PAGE=22f;
    /** 卡片标题。 */
    public static final float CARD_TITLE=17f;
    /** 正文。 */
    public static final float BODY=14f;
    /** 辅助说明。 */
    public static final float NOTE=12f;
    /** 脚注。 */
    public static final float MICRO=11f;

    private VisualStyle(){}

    /** 将输入字号映射到六档标准字号。 */
    public static float size(float requested){
        if(!(requested>0)||Float.isNaN(requested))return BODY;
        if(requested>=30f)return DISPLAY;
        if(requested>=19.5f)return PAGE;
        if(requested>=15f)return CARD_TITLE;
        if(requested>=13f)return BODY;
        if(requested>=11.5f)return NOTE;
        return MICRO;
    }

    public static int dp(Context context,float value){return Math.round(value*context.getResources().getDisplayMetrics().density);}

    public static GradientDrawable rounded(Context context,int color,int radius,boolean border){
        GradientDrawable result=new GradientDrawable();result.setColor(color);result.setCornerRadius(dp(context,radius));
        if(border)result.setStroke(dp(context,1),BORDER);return result;
    }

    /** 首页 hero：暖米 → 浅珊瑚 → 极浅青，三段渐变。 */
    public static GradientDrawable heroBackground(Context context){
        GradientDrawable background=new GradientDrawable(GradientDrawable.Orientation.TL_BR,
                new int[]{0xfffdfbf7,0xfff7efe6,0xffeef4f1});
        background.setCornerRadius(dp(context,20));
        background.setStroke(dp(context,1),0x33d9562f);
        return background;
    }

    /** 评分卡：极浅的暖白纵向渐变，与其它白卡区分。 */
    public static GradientDrawable scoreCardBackground(Context context){
        GradientDrawable background=new GradientDrawable(GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{0xfffbf7f2,0xffffffff});
        background.setCornerRadius(dp(context,18));
        background.setStroke(dp(context,1),BORDER);
        return background;
    }

    /** 浅底附注块。 */
    public static GradientDrawable noteBackground(Context context){
        GradientDrawable background=new GradientDrawable();background.setColor(CARD_TINT);
        background.setCornerRadius(dp(context,12));return background;
    }

    public static AppCompatButton button(Context context,String label,boolean primary){
        AppCompatButton result=new AppCompatButton(context);result.setText(label);result.setAllCaps(false);
        result.setTextSize(BODY);
        result.setTypeface(Typeface.DEFAULT,Typeface.BOLD);result.setGravity(Gravity.CENTER);result.setIncludeFontPadding(false);
        result.setMinWidth(0);result.setMinimumWidth(0);result.setMinHeight(dp(context,48));result.setMinimumHeight(dp(context,48));
        result.setPadding(dp(context,8),dp(context,10),dp(context,8),dp(context,10));
        result.setTextColor(new ColorStateList(new int[][]{new int[]{-android.R.attr.state_enabled},new int[]{}},new int[]{DISABLED,primary?Color.WHITE:ACCENT}));
        androidx.core.view.ViewCompat.setBackgroundTintList(result,null);result.setBackgroundTintList(null);
        StateListDrawable states=new StateListDrawable();
        states.addState(new int[]{-android.R.attr.state_enabled},rounded(context,0xfff2efe9,12,true));
        states.addState(new int[]{},rounded(context,primary?ACCENT:Color.WHITE,12,!primary));
        result.setBackground(new RippleDrawable(ColorStateList.valueOf(primary?0x28ffffff:0x3316324f),states,rounded(context,Color.WHITE,12,false)));
        result.setStateListAnimator(null);result.setElevation(0);result.setLayoutParams(new LinearLayout.LayoutParams(-1,-2));
        return result;
    }

    public static void icon(Context context,AppCompatButton button,int resource,boolean primary){
        Drawable drawable=androidx.appcompat.content.res.AppCompatResources.getDrawable(context,resource);
        if(drawable==null)return;drawable=drawable.mutate();
        drawable.setTintList(new ColorStateList(new int[][]{new int[]{-android.R.attr.state_enabled},new int[]{}},new int[]{DISABLED,primary?Color.WHITE:ACCENT}));
        drawable.setBounds(0,0,dp(context,17),dp(context,17));button.setCompoundDrawablesRelative(drawable,null,null,null);button.setCompoundDrawablePadding(dp(context,5));
    }
}
