
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


constexpr auto face_cascade_path = "C:\\Users\\Beni\\Desktop\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
constexpr auto eye_cascade_path = "C:\\Users\\Beni\\Desktop\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml";

constexpr auto scale = 1;

using namespace std;
using namespace cv;

Mat frame;
Mat grayscale;
Mat Face_Roi_Full;
Mat Face_Roi_Right_Half;
Mat Face_Roi_Left_Half;
Mat eye0_im;
Mat eye1_im;
Mat histImage;
Mat left_pupil_c;
Mat right_pupil_c;
Mat drawing;
Mat drawing1;

CascadeClassifier faceCascade;
CascadeClassifier eyeCascade;

vector <Rect> faces;
static size_t i;


void set_Path() 
    {
		faceCascade.load(face_cascade_path); //beépített haarcascade módszer használatához szükséges útvonal
		eyeCascade.load(eye_cascade_path);
    }

void convert_toGray()
    {
		cvtColor(frame, grayscale, COLOR_BGR2GRAY);  //fekete-fehérré konvertálom a képet, hogy robosztusabb legyen az arcfelismerés
		resize(grayscale, grayscale, Size(grayscale.size().width / scale, grayscale.size().height / scale)); // leveszem a felére (osztom a scale változóval) a képfelbontást, hogy stabilabb fps-t kapjak
    }

void detect_Face() 
    {
		faceCascade.detectMultiScale(grayscale, faces, 1.1, 3, 0, Size(30, 30));  // a haarcascade által felismert arcot a faces vektorban tároljuk el
    }

void detect_left_eye() // bal szem detektálása
    {   
		Rect Crop_Face_Left(faces[i].x, faces[i].y, faces[i].width / 2, faces[i].height / 2); 
		Face_Roi_Left_Half = grayscale(Crop_Face_Left); //körülvágom a grayscale képet az arc bal felső negyedének a koordinátája mentén

		vector <Rect> eye_left;
		eyeCascade.detectMultiScale(Face_Roi_Left_Half, eye_left, 1.1, 3, 0, Size(20, 20));  // a bal szemet csak az arc bal felső negyedében keresem, ezért sokkal pontosabb mintha az egész arcon keresném
		for (size_t k = 0; k < eye_left.size(); k++) {

			Rect eye_0;
			eye_0 = { eye_left[k].x, eye_left[k].y + 14, eye_left[k].width, eye_left[k].height - 21}; // minden egyes framenél körülvágom az arcot mutató képet a szem koordinitáival
			eye0_im = Face_Roi_Left_Half(eye_0);

    }
    }

void detect_right_eye() // jobb szem detektálása, teljesen úgy mint a bal szemet, csak az arc másik oldalán
    {
		Rect Crop_Face_Right(faces[i].x + faces[i].width / 2, faces[i].y, faces[i].width / 2, faces[i].height / 2); 
		Face_Roi_Right_Half = grayscale(Crop_Face_Right);

		vector <Rect> eye_right;
		eyeCascade.detectMultiScale(Face_Roi_Right_Half, eye_right, 1.1, 3, 0, Size(20, 20)); 
		for (size_t l = 0; l < eye_right.size(); l++) {

			Rect eye_1;
			eye_1 = { eye_right[l].x, eye_right[l].y + 14, eye_right[l].width, eye_right[l].height - 21}; 
			eye1_im = Face_Roi_Right_Half(eye_1);
			}
    }

void detect_both_eyes() // mindkét szem detektálása a teljes arc felületén, csak a bemutatás miatt van megvalósítva, mert eléggé inkonszisztens
	{
		Rect Crop_Face(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
		Face_Roi_Full = grayscale(Crop_Face); 

		vector <Rect> eyes;

		eyeCascade.detectMultiScale(Face_Roi_Full, eyes, 1.1, 3, 0, Size(20, 20));
		for (size_t j = 0; j < eyes.size(); j++) {

			rectangle(Face_Roi_Full, Point(eyes[j].x, eyes[j].y), Point(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height), Scalar(255, 255, 255), 1, 1, 0);

		}
	}

void detect_eyes(int num)
    {
		for (i = 0; i < faces.size(); i++) {
			if (num == 2)
			{
				detect_both_eyes();
			}
			if (num == 0)
			{
				detect_left_eye();
			}
			if (num == 1)
			{
				detect_right_eye();
			}        
		}
    }

int calc_hist(Mat src)   // a szem hisztogrammájának kiszámolása, ez alapján fogjuk thresholdolni a
{
	int histSize = 256;
	float range[] = { 0, 256 }; 
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;
	Mat b_hist, g_hist, r_hist;

	calcHist(&src, 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	histImage = { hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0) };

	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	int min_hist = 256;
	for (int i = 1; i < histSize; i++)
	{
		if (b_hist.at<float>(i) > 0) {
			if (i < min_hist) {
				min_hist = i;
			}
		}
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	return min_hist;

}

void detect_pupil_left()
{
	GaussianBlur(eye0_im, left_pupil_c, Size(3, 3), 0);  //a kép szűrése elősegíti a pontosságot
	medianBlur(left_pupil_c, left_pupil_c, 3);

	int thresh = calc_hist(eye0_im);
	calc_hist(eye0_im);
	threshold(left_pupil_c, left_pupil_c, thresh + 11, 255, THRESH_BINARY);  // thresholdolás, azaz a fekete-fehér kép egy bizonyos intenzitás alatt csak fekete, felette pedig csak fehér lesz

	vector<vector<Point> > contours; 
	vector<Vec4i> hierarchy;

	findContours(left_pupil_c, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);  // a thresholdolt képen megkeressük a kontúrokat

	drawing = Mat::zeros(left_pupil_c.size(), CV_8UC3);
	int area = 0;
	int index = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		if ((contours[i][0] != Point(0, 0)))  {
			if (area < contours[i].size())
				index = i;

			Scalar color = Scalar(128);
			drawContours(drawing, contours, (int)i , 128, 2, LINE_8, hierarchy, 0);

		}
	}

	Rect rect = boundingRect(contours[index]);
	Point c_xy;

	c_xy.x = rect.x + rect.width / 2; 
	c_xy.y = rect.y + rect.height / 2;

	circle(eye0_im, c_xy, 0.1, 255, 1);  // kirajzoljuk a kontúrok közepét, amely jelene esetben a pupilla közepe is lesz.

}



void detect_pupil_right()
	{

	GaussianBlur(eye1_im, right_pupil_c, Size(3, 3), 0);
	medianBlur(right_pupil_c, right_pupil_c, 3);

	int thresh = calc_hist(eye1_im);
	threshold(right_pupil_c, right_pupil_c, thresh + 25, 255, THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(right_pupil_c, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	drawing1 = Mat::zeros(right_pupil_c.size(), CV_8UC3);
	int area = 0;
	int index = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		if ((contours[i][0] != Point(0, 0))) {
			if (area < contours[i].size())
				index = i;

			Scalar color = Scalar(128);
			drawContours(drawing1, contours, (int)i, 128, 2, LINE_8, hierarchy, 0);

		}
	}

	Rect rect = boundingRect(contours[index]);
	Point c_xy;

	c_xy.x = rect.x + rect.width / 2;
	c_xy.y = rect.y + rect.height / 2;

	circle(eye1_im, c_xy, 0.1, 255, 1);	

	}



int main() {
    
    set_Path();     

    VideoCapture cap(0);    // ha nem kap jelet a program a webkamerától, akkor lépjen ki
    if (!cap.isOpened()) 
	{
        return -1;
    }

    static int count = -1;

    for (;;) //végtelen ciklus amíg van vétel a webkamerától, képkockánként történik iteráció
    {
        cap >> frame;  // jelenítsen meg minden képkocát
		++count; //képkockák számlálása

		convert_toGray();
 
		if (count <= 10)
		{
			detect_Face();
		}

		if (count > 10) {
			//detect_eyes(2);
			detect_eyes(0);
			detect_eyes(1);
			detect_pupil_left();
			detect_pupil_right();

			resize(eye0_im, eye0_im, Size(eye0_im.size().width * 3, eye0_im.size().height * 3)); //csak a bemutatást kedvéért felnagyítom a két szem képét
			resize(eye1_im, eye1_im, Size(eye1_im.size().width * 3, eye1_im.size().height * 3));
			//resize(drawing, drawing, Size(drawing.size().width * 6, drawing.size().height * 6));
			//resize(left_pupil_c, left_pupil_c, Size(left_pupil_c.size().width * 6, left_pupil_c.size().height * 6));			

			//cv::imshow("Grayscale", grayscale);
			//cv::imshow("Face + eyes", Face_Roi_Full);

			cv::imshow("Left_eye", eye0_im);
			//cv::imshow("Thresholding", left_pupil_c);
			//cv::imshow("contours_left", drawing);
			cv::imshow("Right_eye", eye1_im);
			//cv::imshow("contours_right", drawing1);
			//imshow("calcHist Demo", histImage);
		}

		cv::waitKey(1);    

        }

        cap.release();
        return 0;

}