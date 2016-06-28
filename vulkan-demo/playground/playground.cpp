#include <stdio.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera {
private:
	glm::quat rotation;
	float distance;
	float fov;

public:
	Camera() {
		rotation = glm::quat(1, 0, 0, 0);
		distance = 0.0f;
		fov = 60.0f;
	}

	// Generate camera from Euler angles
	void __generate__(float dist, float pitch, float yaw) {
		float roll = 0.0f;
		rotation = glm::quat(glm::vec3(pitch, yaw, roll));
		distance = dist;
	}

	// Generate camera from lookAt
	void lookat(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 up) {
		rotation = glm::normalize(glm::quat_cast(glm::lookAt(lookfrom, lookat, up)));
		//distance = glm::length(lookfrom - lookat);
		distance = 0.0f;
	}

	glm::mat4 view() {
		glm::mat4 view = glm::mat4_cast(rotation);
		//view = glm::translate()
		return view;
	}
};

void print(glm::mat4 const & Mat0)
{
	printf("mat4(\n");
	printf("\tvec4(%2.3f, %2.3f, %2.3f, %2.3f)\n", Mat0[0][0], Mat0[0][1], Mat0[0][2], Mat0[0][3]);
	printf("\tvec4(%2.3f, %2.3f, %2.3f, %2.3f)\n", Mat0[1][0], Mat0[1][1], Mat0[1][2], Mat0[1][3]);
	printf("\tvec4(%2.3f, %2.3f, %2.3f, %2.3f)\n", Mat0[2][0], Mat0[2][1], Mat0[2][2], Mat0[2][3]);
	printf("\tvec4(%2.3f, %2.3f, %2.3f, %2.3f))\n\n", Mat0[3][0], Mat0[3][1], Mat0[3][2], Mat0[3][3]);
}


int main() {
	Camera C;

	glm::vec3 lookfrom(10, 1, 1);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, -1, 0);

	C.lookat(lookfrom, lookat, up);

	glm::mat4 view = glm::lookAt(lookfrom, lookat, up);

	if (view != C.view()) {
		fprintf(stderr, "Error: Incorrect implementation\n");

		print(view);

		print(C.view());
	}

	system("pause");
	return 0;
}